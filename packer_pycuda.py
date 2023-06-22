import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from string import Template

from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform
from matplotlib.colors import LinearSegmentedColormap
from pycuda.compiler import SourceModule
from tqdm import tqdm

flake_size = 1000


def get_neighbours(col: int, row: int) -> np.array:
    """Return the list of hexagonal neighbours of a cell."""
    # Get the neighbours of the cell
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
__global__ void update_flake_kernel(int *flake, int *new_flake, int *rule) {
    int n_cols = $flake_size;
    int n_rows = $flake_size;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int flake_idx = col * n_rows + row;

    if (col > 0 && col < n_cols - 1 && row > 0 && row < n_rows - 1) {

        int neighbours[6][2] = {
            {col - 1, row},
            {col + 1, row},
            {col, row - 1},
            {col, row + 1},
            {col - 1, row - 1},
            {col + 1, row + 1}
        };

        int frozen_neighbours = 0;
        for (int i = 0; i < 6; i++) {
            int neighbour_col = neighbours[i][0];
            int neighbour_row = neighbours[i][1];
            frozen_neighbours += flake[neighbour_col * n_rows + neighbour_row];
        }

        int should_freeze = rule[frozen_neighbours];

        if (should_freeze) {
            new_flake[flake_idx] = 1;
        } else {
            new_flake[flake_idx] = flake[flake_idx];
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

    flake = np.zeros((flake_size, flake_size)).astype(np.int32)
    # Set the initial state of the flake, with a single frozen cell in the middle
    flake[flake_size // 2, flake_size // 2] = 1
    # Set the rule to use
    rule = np.array([0, 1, 0, 1, 0, 1, 0]).astype(np.int32)

    n_cols, n_rows = flake.shape

    # Define the block and grid sizes
    block_size = (16, 16, 1)
    grid_size = ((n_cols - 1) // block_size[0] + 1, (n_rows - 1) // block_size[1] + 1, 1)

    update_flake_kernel = SourceModule(mod.substitute(flake_size=flake_size)).get_function("update_flake_kernel")

    # Call the kernel function

    # Iterate over the number of steps
    for i in tqdm(range(flake_size // 2)):
        # Copy the flake
        new_flake = flake.copy()
        # Update the flake
        update_flake_kernel(drv.In(flake), drv.Out(new_flake), drv.In(rule), block=block_size, grid=grid_size)
        flake = new_flake
        # Plot the flake
        if i % (flake_size//10) == 0:
            plt.imshow(flake, cmap="gray")
            plt.show()
    render(flake)


sheer_30_degrees = np.matrix([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
squash = np.matrix([[2 / (3 ** 0.5), 0, 0], [0, 1, 0], [0, 0, 1]])


def get_image(snowflake):
    image = snowflake

    # Transform to hexagonal grid by sheering the vertical axis over 30 deg and then squashing vertically
    translate_to_origin = np.matrix([[1, 0, snowflake.shape[0] / 2], [0, 1, snowflake.shape[1] / 2], [0, 0, 1]])
    translate_to_centre = np.matrix([[1, 0, -snowflake.shape[0] / 2], [0, 1, -snowflake.shape[1] / 2], [0, 0, 1]])
    convolution = translate_to_origin * sheer_30_degrees * squash * translate_to_centre
    return affine_transform(image, convolution)


def render_image(snowflake_image, size):
    snowflake_colour_map = LinearSegmentedColormap(
        "",
        {'red': ((0.0, 0.80, 0.80),
                 (0.5, 1.00, 0.80),
                 (1.0, 0.05, 0.05)),

         'green': ((0.0, 0.80, 0.80),
                   (0.5, 1.00, 1.00),
                   (1.0, 0.15, 0.15)),

         'blue': ((0.0, 0.80, 0.80),
                  (0.5, 1.00, 1.00),
                  (1.0, 0.55, 0.55))
         }
    )
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.imshow(snowflake_image, cmap=snowflake_colour_map)
    ax.set_axis_off()
    plt.show()


def render(snowflake, size=12):
    render_image(get_image(snowflake), size=size)


if __name__ == "__main__":
    main()
