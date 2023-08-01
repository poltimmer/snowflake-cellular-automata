import numpy as np
from h5py import File
from numba.cuda.cudadrv.devicearray import DeviceNDArray


def save_buffer_to_hdf5(dset, buffer, i):
    # Write the buffer to the correct position in the dataset
    # The start index is the current step minus the buffer length plus one
    start = i - len(buffer) + 1
    dset[start:start + len(buffer)] = buffer


def save_flake_to_hdf5(flake: np.ndarray, flake_device: DeviceNDArray, step: int, h5_file: File, hash_function='sha256'):
    # Copy data from device to host
    flake_device.copy_to_host(flake)

    # Create a dataset for this step
    dataset = h5_file.create_dataset(f'step_{step}', data=flake)
    dataset.attrs['step'] = step

    # # Compute the hash of the first layer of the flake
    # flake_layer_hash = hashlib.new(hash_function)
    # flake_layer_hash.update(flake[:, :, 0].tobytes())
    # dataset.attrs['flake_layer_hash'] = flake_layer_hash.hexdigest()

    # Compute the size of the crystal (modify this as needed)
    crystal_size = np.sum(flake[:, :, 3])
    dataset.attrs['crystal_size'] = crystal_size
