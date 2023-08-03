import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from string import Template

from pycuda import gpuarray
from pycuda.compiler import SourceModule
from tqdm import tqdm

from utils.plotting import render, plot_flake_masses

n_steps = 15439
n_plots = 20
flake_size = 801
rho = 0.4665013317
kappa = 0.0151229294
beta = 1.1075838293
alpha = 0.1235727185
theta = 0.0946601001
mu = 0.1251353424
gamma = 0.0000604751


def get_neighbors(col: int, row: int) -> np.array:
    """Return the list of hexagonal neighbors of a cell."""
    # Get the neighbors of the cell
    return np.array(
        [
            (col - 1, row),
            (col + 1, row),
            (col, row - 1),
            (col, row + 1),
            (col - 1, row - 1),
            (col + 1, row + 1),
        ]
    )


mod = Template("""
__device__ int n_cols = $flake_size;
__device__ int n_rows = $flake_size;
__device__ float alpha = $alpha;
__device__ float beta = $beta;
__device__ float kappa = $kappa;
__device__ float gamma = $gamma;
__device__ float mu = $mu;
__device__ float theta = $theta;

__device__ void get_neighbors(int col, int row, int (*neighbors)[2]) {
    // Get the hexagonal neighbors of a cell
    neighbors[0][0] = col - 1;
    neighbors[0][1] = row;
    neighbors[1][0] = col + 1;
    neighbors[1][1] = row;
    neighbors[2][0] = col;
    neighbors[2][1] = row - 1;
    neighbors[3][0] = col;
    neighbors[3][1] = row + 1;
    neighbors[4][0] = col - 1;
    neighbors[4][1] = row - 1;
    neighbors[5][0] = col + 1;
    neighbors[5][1] = row + 1;
}

__device__ void diffusion(int col, int row, float *flake, float *new_flake) {
    int flake_idx = (col * n_rows + row) * 4;
    // Get the neighbors of the cell
    int neighbors[6][2];
    get_neighbors(col, row, neighbors);
    int neighbor_count = 0;
    float neighborhood_diffusive_mass = 0;
    for (int i = 0; i < 6; i++) {
        int neighbor_col = neighbors[i][0];
        int neighbor_row = neighbors[i][1];
        neighbor_count += flake[(neighbor_col * n_rows + neighbor_row) * 4];
        neighborhood_diffusive_mass += flake[(neighbor_col * n_rows + neighbor_row) * 4 + 3];
    }
    // Reflective boundary condition + adding the current cell
    neighborhood_diffusive_mass += (neighbor_count + 1) * flake[flake_idx + 3];
    // Update the diffusive mass of the cell, keeping it 0 if the cell is frozen
    new_flake[flake_idx + 3] = neighborhood_diffusive_mass / 7;
}

__device__ void freezing(int col, int row, float *flake, float *new_flake) {
    int flake_idx = (col * n_rows + row) * 4;
    // Get the neighbors of the cell, and check if this is a boundary cell
    int neighbors[6][2];
    get_neighbors(col, row, neighbors);
    bool has_frozen_neighbor = false;
    for (int i = 0; i < 6; i++) {
        int neighbor_col = neighbors[i][0];
        int neighbor_row = neighbors[i][1];
        has_frozen_neighbor = has_frozen_neighbor || (flake[(neighbor_col * n_rows + neighbor_row) * 4] != 0);
    }
    if (!has_frozen_neighbor) {
        return;
    }
    // Given that this is a boundary cell, convert the diffusive mass to crystal mass and boundary mass
    float diffusive_mass = flake[flake_idx + 3];
    new_flake[flake_idx + 1] = flake[flake_idx + 1] + diffusive_mass * (1.0 - kappa);
    new_flake[flake_idx + 2] = flake[flake_idx + 2] + diffusive_mass * kappa;
    new_flake[flake_idx + 3] = 0;
}
    
__device__ void attachment(int col, int row, float *flake, float *new_flake) {
    int flake_idx = (col * n_rows + row) * 4;
    // Get the neighbors of the cell
    int neighbors[6][2];
    get_neighbors(col, row, neighbors);
    int neighbor_count = 0;
    float neighbor_diffusive_mass = 0;
    for (int i = 0; i < 6; i++) {
        int neighbor_col = neighbors[i][0];
        int neighbor_row = neighbors[i][1];
        neighbor_count += flake[(neighbor_col * n_rows + neighbor_row) * 4];
        neighbor_diffusive_mass += flake[(neighbor_col * n_rows + neighbor_row) * 4 + 3];
    }
    
    switch (neighbor_count) {
        case 0:
            // If there are no neighbors, then the cell is not a boundary cell
            return;
        case 1:
        case 2:
            if (flake[flake_idx + 1] >= beta) {
                new_flake[flake_idx] = 1;
                new_flake[flake_idx + 2] = flake[flake_idx + 1];
                new_flake[flake_idx + 1] = 0;
            }
            break;
        case 3:
            if (flake[flake_idx + 1] >= alpha && neighbor_diffusive_mass < theta) {
                new_flake[flake_idx] = 1;
                new_flake[flake_idx + 2] = flake[flake_idx + 1];
                new_flake[flake_idx + 1] = 0;
            }
            break;
        default:
            new_flake[flake_idx + 2] = flake[flake_idx + 1];
            new_flake[flake_idx + 1] = 0;
            new_flake[flake_idx] = 1;
    }
}

__device__ void melting(int col, int row, float *flake, float *new_flake) {
    int flake_idx = (col * n_rows + row) * 4;
    // Get the neighbors of the cell
    float boundary_mass = flake[flake_idx + 1];
    float crystal_mass = flake[flake_idx + 2];
    if (boundary_mass != 0 || crystal_mass != 0) {
        new_flake[flake_idx + 3] = flake[flake_idx + 3] + boundary_mass * mu + crystal_mass * gamma;
        new_flake[flake_idx + 1] = boundary_mass * (1.0 - mu);
        new_flake[flake_idx + 2] = crystal_mass * (1.0 - gamma);
    }
}

__global__ void update(float *flake, float *new_flake, int step) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (flake[(col * n_rows + row) * 4] == 1) {
        // If the cell is frozen, do nothing
        return;
    }
    if (col > 0 && col < n_cols - 1 && row > 0 && row < n_rows - 1) {
        switch (step) {
            case 0:
                diffusion(col, row, flake, new_flake);
                break;
            case 1:
                freezing(col, row, flake, new_flake);
                break;
            case 2:
                attachment(col, row, flake, new_flake);
                break;
            case 3:
                melting(col, row, flake, new_flake);
                break;
            default:
                printf("Invalid step number %d", step);
                break;
        }
    }
}
""")


def main():
    device = drv.Device(0)
    print("Device name:", device.name())
    print("Compute capability:", device.compute_capability())
    print("Total memory:", device.total_memory() / 1024 ** 2, "MB")
    # Print the maximum grid size and block size in each dimension
    print("Max grid size: ({}, {}, {})".format(
        device.get_attribute(drv.device_attribute.MAX_GRID_DIM_X),
        device.get_attribute(drv.device_attribute.MAX_GRID_DIM_Y),
        device.get_attribute(drv.device_attribute.MAX_GRID_DIM_Z)))
    print("Max block size: ({}, {}, {})".format(
        device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X),
        device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Y),
        device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Z)))
    print("Max threads per block:", device.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK))

    flake = np.zeros((flake_size, flake_size, 4)).astype(np.float32)
    # Set the initial state of the flake, with all cells filled with diffusive mass
    flake[:, :, 3].fill(rho)

    # Freeze the centre cell. Set attachment parameter a to 1 and set crystal mass c to 1,
    # and set diffusive mass d to 0.
    flake[flake_size // 2, flake_size // 2, 0] = 1
    flake[flake_size // 2, flake_size // 2, 2] = 1
    flake[flake_size // 2, flake_size // 2, 3] = 0

    n_cols, n_rows, _ = flake.shape

    # Define the block and grid sizes
    block_size = (16, 16, 1)
    grid_size = ((n_cols - 1) // block_size[0] + 1, (n_rows - 1) // block_size[1] + 1, 1)

    mod_filled = mod.substitute(flake_size=flake_size, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa, mu=mu,
                                theta=theta)

    update = SourceModule(mod_filled).get_function("update")

    flake_gpu = gpuarray.to_gpu(flake)
    new_flake_gpu = gpuarray.to_gpu(flake)

    for i in tqdm(range(n_steps)):
        # Update the flake
        update(flake_gpu, new_flake_gpu, np.int32(0), block=block_size, grid=grid_size)
        flake_gpu = new_flake_gpu.copy()
        update(flake_gpu, new_flake_gpu, np.int32(1), block=block_size, grid=grid_size)
        flake_gpu = new_flake_gpu.copy()
        update(flake_gpu, new_flake_gpu, np.int32(2), block=block_size, grid=grid_size)
        flake_gpu = new_flake_gpu.copy()
        update(flake_gpu, new_flake_gpu, np.int32(3), block=block_size, grid=grid_size)
        flake_gpu = new_flake_gpu.copy()

        if i % (n_steps // n_plots) == n_steps // n_plots - 1:
            flake = new_flake_gpu.get()  # Get the data back only for plotting
            plot_flake_masses(flake)

    flake = new_flake_gpu.get()
    render(np.array(flake.tolist()))


if __name__ == "__main__":
    main()
