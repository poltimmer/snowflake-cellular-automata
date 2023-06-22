import math
import logging
import numpy as np
from numba import cuda, int32, float32
from tqdm import tqdm

from utils import render, plot_flake_masses
from numba import types

# logger = logging.getLogger('numba')
# logger.setLevel(logging.DEBUG)

# n_steps = 15439
n_steps = 10000
n_plots = 20
flake_size = 801

rho = 0.5985382874
kappa = 0.0110129997
mu = 0.0384250351
gamma = 0.0000943761
alpha = 0.2859409773
beta = 1.6311921080
theta = 0.0333784865

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
        neighbors[i] = col + offsets[i, 0], row + offsets[i, 1]


@cuda.jit(device=True)
def diffusion(col: int32, row: int32, flake: types.Array(float32, 3, 'A'),
              new_flake: types.Array(float32, 3, 'A')):
    # Create a list to store neighbor coordinates
    neighbors = cuda.local.array((6, 2), int32)

    get_neighbors(col, row, neighbors)
    neighbor_count = 0
    neighborhood_diffusive_mass = 0.0

    for i in range(6):  # Hexagonal grid will always have 6 neighbors
        neighbor = neighbors[i]
        if (0 <= neighbor[0] < flake_size and
                0 <= neighbor[1] < flake_size):
            neighbor_count += flake[neighbor[0], neighbor[1], 0]
            neighborhood_diffusive_mass += flake[neighbor[0], neighbor[1], 3]

    # Reflective boundary condition + adding the current cell
    neighborhood_diffusive_mass += (neighbor_count + 1) * flake[col, row, 3]

    # Update the diffusive mass of the cell, keeping it 0 if the cell is frozen
    new_flake[col, row, 3] = neighborhood_diffusive_mass / 7


@cuda.jit(device=True)
def freezing(col: int32, row: int32, flake: types.Array(float32, 3, 'A'),
             new_flake: types.Array(float32, 3, 'A')):
    neighbors = cuda.local.array((6, 2), int32)
    get_neighbors(col, row, neighbors)

    has_frozen_neighbor = False
    for i in range(neighbors.shape[0]):
        if 0 <= neighbors[i, 0] < flake_size and 0 <= neighbors[i, 1] < flake_size:
            if flake[neighbors[i, 0], neighbors[i, 1], 0] != 0:
                has_frozen_neighbor = True
                break

    if not has_frozen_neighbor:
        return

    diffusive_mass = flake[col, row, 3]
    new_flake[col, row, 1] += diffusive_mass * (1.0 - kappa)
    new_flake[col, row, 2] += diffusive_mass * kappa
    new_flake[col, row, 3] = 0


@cuda.jit(device=True)
def attachment(col: int32, row: int32, flake: types.Array(float32, 3, 'A'),
               new_flake: types.Array(float32, 3, 'A')):
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
            neighbor_count += flake[neighbor[0], neighbor[1], 0]

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
                neighbor_diffusive_mass += flake[neighbor[0], neighbor[1], 3]

    # Given that this is a boundary cell, convert the diffusive mass to crystal mass and boundary mass
    crystal_mass = flake[col, row, 1]
    if (neighbor_count in (1, 2) and crystal_mass >= beta) or \
            (neighbor_count == 3 and crystal_mass >= alpha and neighbor_diffusive_mass < theta) or \
            (neighbor_count > 3):
        new_flake[col, row, 0] = 1
        new_flake[col, row, 1] = 0
        new_flake[col, row, 2] = crystal_mass


@cuda.jit(device=True)
def melting(col: int32, row: int32, flake: types.Array(float32, 3, 'A'),
            new_flake: types.Array(float32, 3, 'A')):
    """
    Perform melting of boundary mass and crystal mass to diffusive mass.
    """
    boundary_mass = flake[col, row, 1]
    crystal_mass = flake[col, row, 2]

    if boundary_mass != 0 or crystal_mass != 0:
        # convert boundary and crystal mass to diffusive mass
        new_flake[col, row, 3] += boundary_mass * mu + crystal_mass * gamma
        # reduce the boundary and crystal mass to by the same amount
        new_flake[col, row, 1] -= boundary_mass * mu
        new_flake[col, row, 2] -= boundary_mass * gamma


@cuda.jit
def update(flake: types.Array(float32, 3, 'A'), new_flake: types.Array(float32, 3, 'A'), step: int32):
    """
    Perform update step on a given cell in the snowflake simulation grid based on the step number.
    """
    # Calculate the column and row index for the current thread
    col, row = cuda.grid(2)

    # Check if the cell is within the flake grid boundaries and is not frozen
    if 0 < col < flake_size - 1 and 0 < row < flake_size - 1 and flake[col, row, 0] != 1:
        if step == 0:
            diffusion(col, row, flake, new_flake)
        elif step == 1:
            freezing(col, row, flake, new_flake)
        elif step == 2:
            attachment(col, row, flake, new_flake)
        elif step == 3:
            melting(col, row, flake, new_flake)


def main():
    flake = np.zeros((flake_size, flake_size, 4)).astype(np.float32)
    flake[:, :, 3].fill(rho)

    flake[flake_size // 2, flake_size // 2, 0] = 1
    flake[flake_size // 2, flake_size // 2, 2] = 1
    flake[flake_size // 2, flake_size // 2, 3] = 0

    new_flake = flake.copy()

    # Transfer data to device
    flake_device = cuda.to_device(flake)
    new_flake_device = cuda.to_device(new_flake)

    n_cols, n_rows, _ = flake.shape

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(flake.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(flake.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for i in tqdm(range(n_steps)):
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 0)
        flake_device[:] = new_flake_device
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 1)
        flake_device[:] = new_flake_device
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 2)
        flake_device[:] = new_flake_device
        update[blockspergrid, threadsperblock](flake_device, new_flake_device, 3)
        flake_device[:] = new_flake_device

        if i % (n_steps // n_plots) == n_steps // n_plots - 1:
            flake_device.copy_to_host(flake)
            flake_crop = flake[flake_size // 2:, flake_size // 2:, :]
            plot_flake_masses(flake_crop)

    flake_device.copy_to_host(flake)
    render(np.array(flake.copy().tolist()))


if __name__ == "__main__":
    main()
