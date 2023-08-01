import math
import os

import h5py
import numpy as np
from numba import cuda, int32, float32, boolean
from numba import types
from tqdm import tqdm

from utils.loading import load
from utils.params import ParamArray, ParamToIdx
from utils.plotting import render, plot_flake_masses, get_flake_filename
from utils.saving import save_flake_to_hdf5

# logger = logging.getLogger('numba')
# logger.setLevel(logging.DEBUG)
simulate_quadrant = True
save_flakes = True
save_dir = 'data/flakes'

# n_steps = 15439
n_steps = 800
n_plots = 20
n_saves = 100
flake_size = 51
batch_size = 128

# rho = 0.5985382874
# kappa = 0.0110129997
# mu = 0.0384250351
# gamma = 0.0000943761
# alpha = 0.2859409773
# beta = 1.6311921080
# theta = 0.0333784865

params: np.array

# rho = 0.4665013317
# kappa = 0.0151229294
# beta = 1.1075838293
# alpha = 0.1235727185
# theta = 0.0946601001
# mu = 0.1251353424
# gamma = 0.0000604751

# Define the relative offsets of the hexagonal neighbors
off = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1]], dtype=np.int32)


@cuda.jit(device=True)
def get_neighbors(col, row, neighbors):
    offsets = cuda.const.array_like(off)  # todo: see if this impacts performance
    # Apply the offsets to the given coordinates
    for i in range(6):
        col_offset, row_offset = offsets[i]
        if simulate_quadrant:
            if col == 0:
                if col_offset == -1:
                    col_offset = 1
                    row_offset += 1
            if row == 0:
                if row_offset == -1:
                    row_offset = 1
                    col_offset += 1
        neighbors[i] = col + col_offset, row + row_offset


@cuda.jit(device=True)
def diffusion(flake_idx: int32, col: int32, row: int32, flake: types.Array(float32, 4, 'A'),
              new_flake: types.Array(float32, 4, 'A')):
    # Create a list to store neighbor coordinates
    neighbors = cuda.local.array((6, 2), int32)

    get_neighbors(col, row, neighbors)
    neighbor_count = 0
    neighborhood_diffusive_mass = 0.0

    for i in range(6):  # Hexagonal grid will always have 6 neighbors
        neighbor = neighbors[i]
        if (0 <= neighbor[0] < flake_size and
                0 <= neighbor[1] < flake_size):
            neighbor_count += flake[flake_idx, neighbor[0], neighbor[1], 0]
            neighborhood_diffusive_mass += flake[flake_idx, neighbor[0], neighbor[1], 3]

    # Reflective boundary condition + adding the current cell
    neighborhood_diffusive_mass += (neighbor_count + 1) * flake[flake_idx, col, row, 3]

    # Update the diffusive mass of the cell, keeping it 0 if the cell is frozen
    new_flake[flake_idx, col, row, 3] = neighborhood_diffusive_mass / 7


@cuda.jit(device=True)
def freezing(flake_idx: int32, col: int32, row: int32, flake: types.Array(float32, 4, 'A'),
             new_flake: types.Array(float32, 4, 'A')):
    neighbors = cuda.local.array((6, 2), int32)
    get_neighbors(col, row, neighbors)

    has_frozen_neighbor = False
    for i in range(neighbors.shape[0]):
        if 0 <= neighbors[i, 0] < flake_size and 0 <= neighbors[i, 1] < flake_size:
            if flake[flake_idx, neighbors[i, 0], neighbors[i, 1], 0] != 0:
                has_frozen_neighbor = True
                break

    if not has_frozen_neighbor:
        return

    diffusive_mass = flake[flake_idx, col, row, 3]
    new_flake[flake_idx, col, row, 1] += diffusive_mass * (1.0 - params[flake_idx, ParamToIdx.KAPPA.value])
    new_flake[flake_idx, col, row, 2] += diffusive_mass * params[flake_idx, ParamToIdx.KAPPA.value]
    new_flake[flake_idx, col, row, 3] = 0


@cuda.jit(device=True)
def attachment(flake_idx: int32, col: int32, row: int32, flake: types.Array(float32, 4, 'A'),
               new_flake: types.Array(float32, 4, 'A'), flake_changed: types.Array(boolean, 1, 'A')):
    """
    Perform attachment of diffusive mass to crystal and boundary mass, based on number of frozen neighbors.
    """
    neighbors = cuda.local.array((6, 2), int32)
    get_neighbors(col, row, neighbors)
    neighbor_count = 0

    for i in range(6):
        neighbor = neighbors[i]
        if (0 <= neighbor[0] < flake_size and
                0 <= neighbor[1] < flake_size):
            neighbor_count += flake[flake_idx, neighbor[0], neighbor[1], 0]

    # If there are no neighbors, then the cell is not a boundary cell
    if neighbor_count == 0:
        return

    # Calculate neighbor_diffusive_mass only if needed
    neighbor_diffusive_mass = 0.0
    if neighbor_count == 3:
        for i in range(6):
            neighbor = neighbors[i]
            if (0 <= neighbor[0] < flake_size and
                    0 <= neighbor[1] < flake_size):
                neighbor_diffusive_mass += flake[flake_idx, neighbor[0], neighbor[1], 3]

    # Given that this is a boundary cell, convert the diffusive mass to crystal mass and boundary mass
    crystal_mass = flake[flake_idx, col, row, 1]
    if (neighbor_count in (1, 2) and crystal_mass >= params[flake_idx, ParamToIdx.BETA.value]) or \
            (neighbor_count == 3 and crystal_mass >= params[
                flake_idx, ParamToIdx.ALPHA.value] and neighbor_diffusive_mass < params[
                 flake_idx, ParamToIdx.THETA.value]) or \
            (neighbor_count > 3):
        new_flake[flake_idx, col, row, 0] = 1
        new_flake[flake_idx, col, row, 1] = 0
        new_flake[flake_idx, col, row, 2] = crystal_mass
        flake_changed[flake_idx] = True


@cuda.jit(device=True)
def melting(flake_idx: int32, col: int32, row: int32, flake: types.Array(float32, 4, 'A'),
            new_flake: types.Array(float32, 4, 'A')):
    """
    Perform melting of boundary mass and crystal mass to diffusive mass.
    """
    boundary_mass = flake[flake_idx, col, row, 1]
    crystal_mass = flake[flake_idx, col, row, 2]

    if boundary_mass != 0 or crystal_mass != 0:
        # convert boundary and crystal mass to diffusive mass
        new_flake[flake_idx, col, row, 3] += boundary_mass * params[flake_idx, ParamToIdx.MU.value] + crystal_mass * \
                                             params[flake_idx, ParamToIdx.GAMMA.value]
        # reduce the boundary and crystal mass to by the same amount
        new_flake[flake_idx, col, row, 1] -= boundary_mass * params[flake_idx, ParamToIdx.MU.value]
        new_flake[flake_idx, col, row, 2] -= boundary_mass * params[flake_idx, ParamToIdx.GAMMA.value]


@cuda.jit
def update(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'), step: int32):
    """
    Perform update step on a given cell in the snowflake simulation grid based on the step number.
    """
    # Calculate the flake idx within the batch, as well as the column and row index for the current thread
    col, row, flake_idx = cuda.grid(3)

    # Check if the cell is within the flake grid boundaries and is not frozen
    if flake[flake_idx, col, row, 0] != 1 and (
            (simulate_quadrant and 0 <= col < flake_size - 1 and 0 <= row < flake_size - 1) or
            (not simulate_quadrant and 0 < col < flake_size - 1 and 0 < row < flake_size - 1)):
        if step == 0:
            diffusion(flake_idx, col, row, flake, new_flake)
        elif step == 1:
            freezing(flake_idx, col, row, flake, new_flake)
        # elif step == 2:
        #     attachment(flake_idx, col, row, flake, new_flake)
        elif step == 3:
            melting(flake_idx, col, row, flake, new_flake)


@cuda.jit
def update_attachment(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
                      flake_changed: types.Array(boolean, 1, 'A')):
    """
    Perform update step on a given cell in the snowflake simulation grid based on the step number.
    """
    # Calculate the flake idx within the batch, as well as the column and row index for the current thread
    col, row, flake_idx = cuda.grid(3)

    # Check if the cell is within the flake grid boundaries and is not frozen
    if flake[flake_idx, col, row, 0] != 1 and (
            (simulate_quadrant and 0 <= col < flake_size - 1 and 0 <= row < flake_size - 1) or
            (not simulate_quadrant and 0 < col < flake_size - 1 and 0 < row < flake_size - 1)):
        attachment(flake_idx, col, row, flake, new_flake, flake_changed)


def main():
    data_array = load()
    param_list = ParamArray(data_array, batch_size=batch_size)
    global params
    for batch in param_list:
        params = batch
        simulate()


def simulate():
    count = 0
    flake = np.zeros((batch_size, flake_size, flake_size, 4)).astype(np.float32)
    rho_array = params[:, ParamToIdx.RHO.value]
    rho_reshaped = rho_array.reshape((-1, 1, 1))
    flake[:, :, :, 3] = rho_reshaped

    if simulate_quadrant:
        flake[:, 0, 0, 0] = 1
        flake[:, 0, 0, 2] = 1
        flake[:, 0, 0, 3] = 0
    else:
        flake[:, flake_size // 2, flake_size // 2, 0] = 1
        flake[:, flake_size // 2, flake_size // 2, 2] = 1
        flake[:, flake_size // 2, flake_size // 2, 3] = 0

    new_flake = flake.copy()
    previous_flake_sum = np.sum(flake[:, :, :, 0], axis=(1, 2))

    # Transfer data to device
    flake_device = cuda.to_device(flake)
    new_flake_device = cuda.to_device(new_flake)

    # _, n_cols, n_rows, _ = flake.shape

    threadsperblock = (16, 16, 1)
    blockspergrid_x = math.ceil(flake_size / threadsperblock[0])
    blockspergrid_y = math.ceil(flake_size / threadsperblock[1])
    blockspergrid_z = math.ceil(batch_size / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    buffer = []

    # Open HDF5 file for writing
    h5_files = []
    if save_flakes:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(batch_size):
            filename = get_flake_filename(*params[i, :])
            h5_file = h5py.File(f"./{save_dir}/{filename}_{i}.h5", 'w')
            # Write simulation parameters as file-level attributes
            h5_file.attrs['rho'] = params[0, ParamToIdx.RHO.value]
            h5_file.attrs['kappa'] = params[0, ParamToIdx.KAPPA.value]
            h5_file.attrs['mu'] = params[0, ParamToIdx.MU.value]
            h5_file.attrs['gamma'] = params[0, ParamToIdx.GAMMA.value]
            h5_file.attrs['alpha'] = params[0, ParamToIdx.ALPHA.value]
            h5_file.attrs['beta'] = params[0, ParamToIdx.BETA.value]
            h5_file.attrs['theta'] = params[0, ParamToIdx.THETA.value]
            h5_files.append(h5_file)

    flake_sum = np.empty_like(flake_device[:, :, :, 0])
    flake_changed = np.zeros((batch_size,)).astype(np.bool_)
    zero_array_device = cuda.to_device(flake_changed)
    flake_changed_device = cuda.to_device(flake_changed)

    for step in tqdm(range(n_steps)):
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 0)
        flake_device[:] = new_flake_device
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 1)
        flake_device[:] = new_flake_device
        update_attachment[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_changed_device)
        flake_device[:] = new_flake_device
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 3)
        flake_device[:] = new_flake_device

        if step % (n_steps // n_plots) == n_steps // n_plots - 1:
            flake_device.copy_to_host(flake)
            # flake_crop = flake[:, flake_size // 2:, flake_size // 2:, :]
            plot_flake_masses(flake[0])

        # flake_sum = np.sum(flake_device[:, :, :, 0], axis=(1, 2))
        # sum_reduce_2D[batch_size, 1](flake_device[:, :, :, 0], flake_sum)
        # flake_sum = []
        # for i in range(batch_size):
        #     flake_sum.append(sum_reduce(np.ascontiguousarray(flake_device[i, :, :, 0]).ravel()))
        if np.any(flake_changed_device):
            print(step)
        for (idx,) in np.argwhere(flake_changed_device):
            if save_flakes:
                save_flake_to_hdf5(flake[idx], flake_device[idx], step, h5_files[idx])
        #         # flake_device.copy_to_host(flake)
        #         # buffer.append(flake.copy())
        #         # if step % (n_steps // n_saves) == n_steps // n_saves - 1:
        #         #     save_buffer_to_hdf5(dset, buffer, step)
        #         #     buffer = []
        # previous_flake_sum = flake_sum
        flake_changed_device[:] = zero_array_device
    flake_device.copy_to_host(flake)
    render(np.array(flake.copy().tolist())[0])
    print(count)

    if save_flakes:
        h5_file.close()


if __name__ == "__main__":
    main()
