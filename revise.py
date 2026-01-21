# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:44:17 2023

@author: user
"""

import numpy as np

def revise2d(target, mask_array, miss_target=[0, 1, 14, 15, 16, 24]):
    """

    :param target:     要重新分割的label ，
    :param mask_array: CMB、WMH 的mask array
    :param miss_target: neighborhood 忽略的 label
    :return:  array
    """

    kk = 3
    temp_array = mask_array.copy()
    while np.any(temp_array == target):
        kk += 1
        window_size = kk
        half_window = window_size // 2
        temp_array_index = np.argwhere(temp_array == target)
        for row in temp_array_index:
            center_x = row[0]
            center_y = row[1]
            center_z = row[2]
            neighborhood = mask_array[center_x - half_window:center_x + half_window + 1,
                           center_y - half_window:center_y + half_window + 1,
                           center_z]
            for i in miss_target:
                neighborhood = np.where(neighborhood == i, np.nan, neighborhood)
            unique, counts = np.unique(neighborhood, return_counts=True)
            if len(unique) > 0:
                if np.isnan(unique[counts.argmax()]):
                    pass
                else:
                    temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
    return temp_array


def revise3d(target, mask_array, miss_target=[0, 1, 14, 15, 16, 24]):
    """

    :param target:     要重新分割的label ，
    :param mask_array: CMB、WMH 的mask array
    :param miss_target: neighborhood 忽略的 label
    :return:  array
    """

    kk = 3
    temp_array = mask_array.copy()
    while np.any(temp_array == target):
        kk += 1
        window_size = kk
        half_window = window_size // 2
        temp_array_index = np.argwhere(temp_array == target)
        for row in temp_array_index:
            center_x = row[0]
            center_y = row[1]
            center_z = row[2]
            neighborhood = mask_array[center_x - half_window:center_x + half_window + 1,
                           center_y - half_window:center_y + half_window + 1,
                           center_z - half_window:center_z + half_window + 1, ]
            for i in miss_target:
                neighborhood = np.where(neighborhood == i, np.nan, neighborhood)
            unique, counts = np.unique(neighborhood, return_counts=True)
            if len(unique) > 0:
                if np.isnan(unique[counts.argmax()]):
                    pass
                else:
                    temp_array[center_x, center_y, center_z] = unique[counts.argmax()]
    return temp_array


if __name__ == '__main__':
    synthseg_array[(synthseg33_array_left_mask & synthseg_array_right_mask)] = 100
    revise3d(100, synthseg_array, miss_target=[0, 1, 14, 15, 16, 24])

    #基本上是這樣。