import json
import math
import os
import random
import shutil
import tempfile

import h5py
import numpy as np
from numba import cuda, int32, float32, boolean, uint64
from numba import types
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states
from tqdm import tqdm

from utils.loading import load
from utils.params import ParamArray, ParamToIdx, ParamList
from utils.plotting import render, plot_flake_masses, get_flake_filename
from utils.saving import save_flake_to_hdf5
from argparse import ArgumentParser

params: np.array
temp_dir = tempfile.mkdtemp()

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

    # if col == 10 and row == 10:
    #     # num = params[flake_idx, ParamToIdx.GAMMA.value]
    #     # integral_part = int(num)
    #     # fractional_part = int((num - integral_part) * 1e15)  # adjust precision as needed
    #     # print(integral_part, fractional_part)
    #     print('rho', params[flake_idx, ParamToIdx.RHO.value])
    #     print('kappa', params[flake_idx, ParamToIdx.KAPPA.value])
    #     print('mu', params[flake_idx, ParamToIdx.MU.value])
    #     print('gamma', params[flake_idx, ParamToIdx.GAMMA.value])
    #     print('alpha', params[flake_idx, ParamToIdx.ALPHA.value])
    #     print('beta', params[flake_idx, ParamToIdx.BETA.value])
    #     print('theta', params[flake_idx, ParamToIdx.THETA.value])

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
    do_freeze = False
    if neighbor_count in (1, 2):
        if crystal_mass >= params[flake_idx, ParamToIdx.BETA.value]:
            do_freeze = True
    if neighbor_count == 3:
        if crystal_mass >= 1.0 or (crystal_mass >= params[flake_idx, ParamToIdx.ALPHA.value]
                                   and neighbor_diffusive_mass < params[flake_idx, ParamToIdx.THETA.value]):
            do_freeze = True
    if neighbor_count >= 4:
        do_freeze = True
    if do_freeze:
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
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for i in range(batch_size):
            dir_name = get_flake_filename(*params[i, :])
            if not os.path.exists(f"{temp_dir}/{dir_name}"):
                os.makedirs(f"{temp_dir}/{dir_name}")
            # flake_number = batch_count * batch_size + i
            flake_number = i % n_repeat
            filename = f"flake_{flake_number}"
            h5_file = h5py.File(f"{temp_dir}/{dir_name}/{filename}.h5", 'w')
            # Write simulation parameters as file-level attributes
            h5_file.attrs['rho'] = params[i, ParamToIdx.RHO.value]
            h5_file.attrs['kappa'] = params[i, ParamToIdx.KAPPA.value]
            h5_file.attrs['mu'] = params[i, ParamToIdx.MU.value]
            h5_file.attrs['gamma'] = params[i, ParamToIdx.GAMMA.value]
            h5_file.attrs['alpha'] = params[i, ParamToIdx.ALPHA.value]
            h5_file.attrs['beta'] = params[i, ParamToIdx.BETA.value]
            h5_file.attrs['theta'] = params[i, ParamToIdx.THETA.value]
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
            2] * blockspergrid_x * blockspergrid_y * blockspergrid_z, seed=random.random() * 2 ** 31)

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

        if plot and n_plots != 0 and step % (max_steps // n_plots) == max_steps // n_plots - 1:
            flake_device.copy_to_host(flake)
            plot_flake_masses(flake[0])

        if save_flakes:
            for (idx,) in np.argwhere(flake_changed_device):
                save_flake_to_hdf5(flake[idx], flake_device[idx], step, h5_files[idx])

        flake_changed_device[:] = zero_array_device
    flake_device.copy_to_host(flake)
    if plot:
        plot_flake_masses(flake[0])
        # render(np.array(flake.copy().tolist())[0])

    if save_flakes:
        for i in range(batch_size):
            h5_files[i].create_dataset('flake_final', data=flake[i], compression='gzip', compression_opts=9)
            # h5_files[i].create_dataset('rng_states_final', data=rng_states, compression='gzip', compression_opts=9)
            h5_files[i].close()

    # Delete the files that are not finished (flake_stop_device is false)
    for (idx,) in np.argwhere(~flake_stop_device):
        dir_name = get_flake_filename(*params[idx, :])
        flake_number = idx % n_repeat
        filename = f"flake_{flake_number}"
        os.remove(f"{temp_dir}/{dir_name}/{filename}.h5")
        # Delete the directory if it's empty
        if not os.listdir(f"{temp_dir}/{dir_name}"):
            os.rmdir(f"{temp_dir}/{dir_name}")


def main():
    SEED = 45

    n_repeat = 8
    free_params = [ParamToIdx.RHO.value, ParamToIdx.THETA.value]

    generate_single = False
    generate_single_index = 4
    n_batches = 4
    batch_count = 0

    quadrant_only_simulation = True
    save_flakes = True
    if generate_single:
        save_dir = f'data/single_flakes_type_{generate_single_index}'
    else:
        save_dir = f'data/flakes_{SEED}'

    max_steps = 10000
    plot = True
    n_plots = 0
    flake_size = 64
    flake_stop_margin = 5
    batch_size = 256

    # sigma = 1e-5
    sigma = 5e-5
    # sigma = 1e-4
    # sigma = 1e-8

    parser = ArgumentParser(description="Gravner-Griffeath snowflake simulation")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_repeat", type=int, default=8, help="Number of times to repeat each parameter set")
    parser.add_argument("--generate_single", action="store_true", help="Generate a single snowflake")
    parser.add_argument("--gen_single_index", type=int, default=4, help="Index of the single snowflake to generate")
    parser.add_argument("--gen_single_n_batches", type=int, default=4,
                        help="Number of batches to generate with single snowflake generation")
    parser.add_argument("--quadrant_only_simulation", default=True, type=bool, help="Only simulate a quadrant")
    parser.add_argument("--save_flakes", default=True, type=bool, help="Save the flakes to HDF5 files")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of simulation steps")
    parser.add_argument("--plot", default=True, type=bool, help="Plot the snowflake")
    parser.add_argument("--n_plots", type=int, default=0, help="Number of plots to generate")
    parser.add_argument("--flake_size", type=int, default=64, help="Size of the snowflake grid")
    parser.add_argument("--flake_stop_margin", type=int, default=5,
                        help="Margin around the flake to stop the simulation")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--sigma", type=float, default=5e-5, help="Noise parameter")

    parser.add_argument("-c", "--config", type=str, default=None, help="Configuration file")

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as file:
            config_args = yaml.safe_load(file)
        for key, value in config_args.items():
            setattr(args, key, value)

    data_array = load()
    random.seed(1)
    np.random.seed(1)
    # param_list = ParamArray(data_array, batch_size=batch_size, randomize=True)
    with open('./flakes.json', 'r') as f:
        data_dict = json.load(f)
    param_list = ParamList(data_dict, batch_size=batch_size, randomize=True, n_repeat=n_repeat)
    fixed_params = param_list[31:32].repeat(batch_size, axis=0)
    # free_params = [ParamToIdx.ALPHA.value, ParamToIdx.BETA.value, ParamToIdx.GAMMA.value, ParamToIdx.KAPPA.value,
    #                ParamToIdx.MU.value, ParamToIdx.RHO.value, ParamToIdx.THETA.value]
    random.seed(SEED)
    np.random.seed(SEED)
    global params
    global batch_count
    if generate_single:
        params = np.array([param_list[generate_single_index]] * batch_size)
        for i in tqdm(range(n_batches)):
            simulate()
            batch_count += 1
    else:
        for batch in tqdm(param_list):
            if len(batch) < batch_size or batch_count >= 20:
                break
            # if batch_count < 30:
            #     batch_count += 1
            #     continue
            params = fixed_params
            params[:, free_params] = batch[:, free_params]
            simulate()
            batch_count += 1

    shutil.move(temp_dir, save_dir)


if __name__ == "__main__":
    main()
