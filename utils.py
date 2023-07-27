import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import affine_transform
import hashlib

sheer_30_degrees = np.matrix([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
squash = np.matrix([[2 / (3 ** 0.5), 0, 0], [0, 1, 0], [0, 0, 1]])

def get_flake_filename(rho, kappa, mu, gamma, alpha, beta, theta):
    """
    Generate a short, deterministic filename based on the parameters of the snowflake.
    """
    # Concatenate parameters into a single string
    params_str = f"{rho}-{kappa}-{mu}-{gamma}-{alpha}-{beta}-{theta}"
    # Create a SHA-256 hash of the string
    params_hash = hashlib.sha256(params_str.encode()).hexdigest()
    # Use the first 10 characters of the hash for the filename
    filename = f"flake_{params_hash[:10]}.hdf5"
    return filename

def get_image(snowflake):
    if len(snowflake.shape) == 3:
        background = np.invert(snowflake[:,:,0].astype(bool)) * snowflake[:,:,3] / np.max(snowflake[:,:,3])
        flake = snowflake[:,:,0] * (snowflake[:,:,2] / np.max(snowflake[:,:,2]))
        image = flake - background
    else:
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


def plot_flake_masses(flake):
    fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(10, 10))
    # select only the top-right quadrant of the flake
    axs[0, 0].imshow(flake[:, :, 3], origin='lower', clim=(0.0, 1.0))
    axs[0, 1].imshow(flake[:, :, 1], origin='lower', clim=(0.0, 1.0))
    axs[1, 0].imshow(flake[:, :, 2], origin='lower', clim=(0.0, 1.0))
    i = axs[1, 1].imshow(flake[:, :, 0], origin='lower', clim=(0.0, 1.0))
    axs[0, 0].set_title("Vapour")
    axs[0, 1].set_title("Boundary mass")
    axs[1, 0].set_title("Ice")
    axs[1, 1].set_title("Attachment")
    cb = fig.colorbar(i, ax=axs)
    plt.show()
