import numpy as np
from numba import cuda
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import render

flake_size = 1000


@cuda.jit
def update_flake_kernel(flake, new_flake, rule):
    n_cols = flake.shape[0]
    n_rows = flake.shape[1]

    col, row = cuda.grid(2)

    if col >= n_cols or row >= n_rows:
        return

    if 0 < col < n_cols - 1 and 0 < row < n_rows - 1:

        frozen_neighbours = 0
        # Directly iterate over the neighbors
        for dcol, drow in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
                           (1, 1)]:  # Exclude the cell itself and top-right and bottom-left
            neighbour_col = col + dcol
            neighbour_row = row + drow
            frozen_neighbours += flake[neighbour_col, neighbour_row]

        should_freeze = rule[frozen_neighbours]

        if should_freeze:
            new_flake[col, row] = 1
        else:
            new_flake[col, row] = flake[col, row]


def main():
    # Set the initial state of the flake, with a single frozen cell in the middle
    flake = np.zeros((flake_size, flake_size), dtype=np.int32)
    flake[flake_size // 2, flake_size // 2] = 1

    # Set the rule to use
    rule = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

    # Define the block and grid sizes
    threadsperblock = (16, 16)
    blockspergrid_x = np.ceil(flake.shape[0] / threadsperblock[0]).astype(int)
    blockspergrid_y = np.ceil(flake.shape[1] / threadsperblock[1]).astype(int)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Move data to the device
    flake_device = cuda.to_device(flake)
    rule_device = cuda.to_device(rule)
    new_flake_device = cuda.device_array_like(flake_device)

    # Iterate over the number of steps
    for i in tqdm(range(flake_size // 2)):
        # Call the kernel function
        update_flake_kernel[blockspergrid, threadsperblock](flake_device, new_flake_device, rule_device)

        # Swap flake_device and new_flake_device
        flake_device, new_flake_device = new_flake_device, flake_device

        # Plot the flake
        if i % (flake_size // 10) == 0:
            flake = flake_device.copy_to_host()
            plt.imshow(flake, cmap="gray")
            plt.show()

    flake = flake_device.copy_to_host()
    render(flake)


if __name__ == "__main__":
    main()
