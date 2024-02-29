import json

import numpy as np

# Parameter mapping
param_name_dict = {
    'r': 'rho',
    'k': 'kappa',
    'm': 'mu',
    'g': 'gamma',
    'a': 'alpha',
    'b': 'beta',
    't': 'theta',
}


def load():
    # Load JSON data
    with open('./flakes.json', 'r') as f:
        data = json.load(f)

    # Initialize data list and param-to-index dictionary
    data_list = []

    # Iterate over flakes
    for flake in data['flakes']:
        params = []
        for key, name in param_name_dict.items():
            value = flake['params'][key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            params.append(float(value))
        data_list.append(params)
    return np.array(data_list)
    #     data_list.append((flake['params']['j'][0], params))
    #
    # data_list.sort(key=lambda x: x[0], reverse=True)
    #
    # # Convert data list to numpy array
    # return np.array([x[1] for x in data_list])


if __name__ == "__main__":
    data_array = load()
    print(data_array.shape)
    print(data_array[0])
    str_arr = np.vectorize(str)(data_array)
    num_decimals = np.vectorize(lambda x: len(x.split('.')[-1]))(str_arr)
    max_decimals = np.max(num_decimals)
    print(max_decimals)
