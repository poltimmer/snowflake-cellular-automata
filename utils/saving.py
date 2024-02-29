import numpy as np
from h5py import File
from numba.cuda.cudadrv.devicearray import DeviceNDArray


def save_buffer_to_hdf5(dset, buffer, i):
    # Write the buffer to the correct position in the dataset
    # The start index is the current step minus the buffer length plus one
    start = i - len(buffer) + 1
    dset[start:start + len(buffer)] = buffer


def save_flake_to_hdf5(flake: np.ndarray, flake_device: DeviceNDArray, step: int, h5_file: File):
    # Copy data from device to host
    flake_device.copy_to_host(flake)

    dt = np.dtype([('ice', np.float32), ('attachment', np.bool_)])
    data = np.zeros((flake.shape[0], flake.shape[1]), dtype=dt)
    data['ice'] = flake[:, :, 2]
    data['attachment'] = flake[:, :, 0].astype(np.bool_)

    # Create a dataset for this step
    dataset = h5_file.create_dataset(f'step_{step}', data=data, compression='gzip', compression_opts=9)
    dataset.attrs['step'] = step
