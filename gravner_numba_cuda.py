import math
import os
from random import random, seed

import h5py
import numpy as np
from numba import cuda, int32, float32, boolean, uint64
from numba import types
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
from tqdm import tqdm

from utils.loading import load
from utils.params import ParamArray, ParamToIdx
from utils.plotting import render, plot_flake_masses, get_flake_filename
from utils.saving import save_flake_to_hdf5

quadrant_only_simulation = True
save_flakes = True
save_dir = 'data/flakes'

max_steps = 5000
plot = False
n_plots = 10
flake_size = 64
flake_stop_margin = 5
batch_size = 256

sigma = 5e-5

params: np.array

# Define the relative offsets of the hexagonal neighbors
off = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1]], dtype=np.int32)


@cuda.jit(device=True)
def get_neighbors(col, row, neighbors):
    offsets = cuda.const.array_like(off)  # todo: see if this impacts performance
    # Apply the offsets to the given coordinates
    for i in range(6):
        col_offset, row_offset = offsets[i]
        if quadrant_only_simulation:
            if col == 0:
                if col_offset == -1:
                    col_offset = 1
                    row_offset += 1
            if row == 0:
                if row_offset == -1:
                    row_offset = 1
                    col_offset += 1
        neighbors[i] = col + col_offset, row + row_offset


@cuda.jit
def diffusion(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
              flake_stop: types.Array(boolean, 1, 'A')):
    col, row, flake_idx = cuda.grid(3)
    if not is_live_cell(flake_idx, col, row, flake, flake_stop):
        return

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


@cuda.jit
def freezing(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
             flake_stop: types.Array(boolean, 1, 'A')):
    """
    Freezing rule:
    If a cell is unfrozen and has a frozen neighbor, it freezes with probability kappa.
    """
    col, row, flake_idx = cuda.grid(3)
    if not is_live_cell(flake_idx, col, row, flake, flake_stop):
        return

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


@cuda.jit
def attachment(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
               flake_changed: types.Array(boolean, 1, 'A'), flake_stop: types.Array(boolean, 1, 'A')):
    """
    Perform attachment of diffusive mass to crystal and boundary mass, based on number of frozen neighbors.
    """
    col, row, flake_idx = cuda.grid(3)
    if not is_live_cell(flake_idx, col, row, flake, flake_stop):
        return

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
        if col == flake_size - flake_stop_margin:
            flake_stop[flake_idx] = True


@cuda.jit
def melting(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
            flake_stop: types.Array(boolean, 1, 'A')):
    """
    Perform melting of boundary mass and crystal mass to diffusive mass.
    """
    col, row, flake_idx = cuda.grid(3)
    if not is_live_cell(flake_idx, col, row, flake, flake_stop):
        return

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
def noise(flake: types.Array(float32, 4, 'A'), new_flake: types.Array(float32, 4, 'A'),
          flake_stop: types.Array(boolean, 1, 'A'), rng_states: types.Array(uint64, 1, 'A')):
    """
    Perform noise addition to diffusive mass.
    Noise is added by adding or subtracting sigma * diffusive_mass, where the sign is chosen randomly with equal
    probability.
    This kernel is deterministic based on the random seed.
    """
    col, row, flake_idx = cuda.grid(3)
    if not is_live_cell(flake_idx, col, row, flake, flake_stop):
        return

    diffusive_mass = flake[flake_idx, col, row, 3]
    if diffusive_mass != 0:
        # Generate a random float and check if it's greater than 0.5
        random_value = xoroshiro128p_uniform_float32(rng_states,
                                                     flake_idx * flake.shape[1] * flake.shape[2] +
                                                     col * flake.shape[2] + row)
        sign = 1 if random_value > 0.5 else -1
        new_flake[flake_idx, col, row, 3] += diffusive_mass * sigma * sign


@cuda.jit(device=True)
def is_live_cell(flake_idx: int32, col: int32, row: int32, flake: types.Array(float32, 4, 'A'),
                 flake_stop: types.Array(boolean, 1, 'A')) -> boolean:
    """
    Check if the cell is within the flake grid boundaries and is not frozen.
    """
    return not flake_stop[flake_idx] and flake[flake_idx, col, row, 0] != 1 and (
            (quadrant_only_simulation and 0 <= col < flake_size - 1 and 0 <= row < flake_size - 1) or
            (not quadrant_only_simulation and 0 < col < flake_size - 1 and 0 < row < flake_size - 1))


def simulate():
    flake = np.zeros((batch_size, flake_size, flake_size, 4)).astype(np.float32)
    rho_array = params[:, ParamToIdx.RHO.value]
    rho_reshaped = rho_array.reshape((-1, 1, 1))
    flake[:, :, :, 3] = rho_reshaped

    if quadrant_only_simulation:
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

    threadsperblock = (16, 16, 4)
    blockspergrid_x = math.ceil(flake_size / threadsperblock[0])
    blockspergrid_y = math.ceil(flake_size / threadsperblock[1])
    blockspergrid_z = math.ceil(batch_size / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Open HDF5 file for writing
    h5_files = []
    if save_flakes:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(batch_size):
            filename = get_flake_filename(*params[i, :])
            h5_file = h5py.File(f"./{save_dir}/{filename}.h5", 'w')
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
    flake_stop = np.zeros((batch_size,)).astype(np.bool_)
    flake_stop_device = cuda.to_device(flake_stop)

    rng_states: types.Array(uint64, 1, 'A') = None

    if sigma > 0:
        rng_states = create_xoroshiro128p_states(threadsperblock[0] * threadsperblock[1] * threadsperblock[
            2] * blockspergrid_x * blockspergrid_y * blockspergrid_z, seed=1)

    for step in tqdm(range(max_steps), desc="Simulation steps", leave=False):
        diffusion[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_stop_device)
        flake_device[:] = new_flake_device
        freezing[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_stop_device)
        flake_device[:] = new_flake_device
        attachment[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_changed_device,
                                                   flake_stop_device)
        flake_device[:] = new_flake_device
        melting[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_stop_device)
        flake_device[:] = new_flake_device
        if np.all(flake_stop_device):
            print(f"Stopped at step {step}")
            break
        if sigma > 0:
            noise[blockspergrid, threadsperblock](flake_device, new_flake_device, flake_stop_device, rng_states)
            flake_device[:] = new_flake_device

        if plot and step % (max_steps // n_plots) == max_steps // n_plots - 1:
            flake_device.copy_to_host(flake)
            plot_flake_masses(flake[0])

        for (idx,) in np.argwhere(flake_changed_device):
            if save_flakes:
                save_flake_to_hdf5(flake[idx], flake_device[idx], step, h5_files[idx])

        flake_changed_device[:] = zero_array_device
    flake_device.copy_to_host(flake)
    if plot:
        render(np.array(flake.copy().tolist())[0])

    if save_flakes:
        for i in range(batch_size):
            h5_files[i].create_dataset('flake_final', data=flake[i], compression='gzip', compression_opts=9)
            # h5_files[i].create_dataset('rng_states_final', data=rng_states, compression='gzip', compression_opts=9)
            h5_files[i].close()


def main():
    data_array = load()
    seed(1)
    np.random.seed(1)
    param_list = ParamArray(data_array, batch_size=batch_size, randomize=True)
    global params
    with tqdm(total=len(param_list)) as pbar:
        for batch in param_list:
            if len(batch) < batch_size:
                break
            params = batch
            simulate()
            pbar.update(len(batch))


if __name__ == "__main__":
    main()
