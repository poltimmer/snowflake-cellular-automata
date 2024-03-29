from collections import OrderedDict
from enum import Enum
from typing import Iterator

import numpy as np


class ParamList:
    def __init__(self, data_dict, batch_size=1, randomize=False, n_repeat=1):
        self.param_keys = ['n', 'a', 'r', 'm', 'x', 's', 'j', 'y', 'k', 'g', 'b', 'z', 'i', 't']
        self.param_name_dict = {
            'r': 'rho',
            'k': 'kappa',
            'm': 'mu',
            'g': 'gamma',
            'a': 'alpha',
            'b': 'beta',
            't': 'theta',
        }
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.n_repeat = n_repeat
        # sort by data_dict['flakes']['params']['j'][0]
        if randomize:
            np.random.shuffle(self.data_dict['flakes'])
        else:
            self.data_dict['flakes'].sort(key=lambda x: x['params']['j'][0], reverse=True)

    def __len__(self):
        return len(self.data_dict['flakes']) * self.n_repeat // self.batch_size

    def __getitem__(self, item):
        if isinstance(item, slice):
            return np.array([self.__get_single_item(i) for i in
                             range(*item.indices(len(self.data_dict['flakes'] * self.n_repeat)))])

        return self.__get_single_item(item)

    def __get_single_item(self, idx):
        flake = self.data_dict['flakes'][idx // self.n_repeat]
        # flake = next((flake for flake in self.data_dict['flakes'] if flake['name'] == 'lianra'), None)

        params = []
        for key, name in self.param_name_dict.items():
            value = flake['params'][key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            params.append(float(value))
        #     print(f"{name}: {value}")
        # print(f"name: {flake['name']}")

        return np.array(params)

    def __iter__(self):
        for i in range(0, len(self.data_dict['flakes']) * self.n_repeat, self.batch_size):
            yield np.array(self[i: i + self.batch_size])


class ParamToIdx(Enum):
    RHO = 0
    KAPPA = 1
    MU = 2
    GAMMA = 3
    ALPHA = 4
    BETA = 5
    THETA = 6


class ParamArray:
    def __init__(self, data_array: np.array, batch_size: int = 1, randomize: bool = False):
        self.data_array: np.array = data_array
        self.batch_size: int = batch_size
        self.randomize: bool = randomize
        if self.randomize:
            np.random.shuffle(self.data_array)

    def __len__(self) -> int:
        return len(self.data_array)

    def __getitem__(self, item) -> np.array:
        return self.data_array[item]

    def __iter__(self) -> Iterator[np.array]:
        for i in range(0, len(self), self.batch_size):
            yield self[i: i + self.batch_size]


def batcher(data, batch_size) -> Iterator[np.array]:
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]
