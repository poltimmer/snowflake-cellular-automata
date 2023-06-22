from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import affine_transform
from tqdm import tqdm

grid_size = 100


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


def update_grid(grid: np.array, rule: List[int]) -> np.array:
    """Updates the grid according to the packard snowflake algorithm.

    Args:
        grid: The grid to update.
        rule: The rule to apply. Is a list of 6 integers, where an integer in the list
            is the number of hexagonal neighbours a cell must have to freeze.

    Returns:
        The updated grid.
    """
    # Create a copy of the grid
    new_grid = grid.copy()
    # Iterate over the grid
    for col in range(1, grid.shape[0] - 1):
        for row in range(1, grid.shape[1] - 1):
            # Get the neighbours of the current cell
            neighbours = get_neighbours(col, row)
            # Count the number of frozen neighbours
            frozen_neighbours = np.sum(grid[neighbours[:, 0], neighbours[:, 1]])
            # Check if the cell should freeze
            if frozen_neighbours in rule:
                new_grid[col, row] = 1
    return new_grid


def main():
    grid = np.zeros((grid_size, grid_size))
    # Set the initial state of the grid, with a single frozen cell in the middle
    grid[grid_size // 2, grid_size // 2] = 1
    # Set the rule to use
    rule = [1, 3, 5]
    # Iterate over the number of steps
    for i in tqdm(range(grid_size // 2)):
        # Update the grid
        grid = update_grid(grid, rule)
        # Plot the grid
        if i % 5 == 0:
            plt.imshow(grid, cmap="gray")
            plt.show()
    render(grid)


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
