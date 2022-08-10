import numpy as np
import matplotlib.pyplot as plt

def minmax(input, gyro_max=4.2, acc_max=16.8):
    gyro = input.loc[:, ['gyro_x', 'gyro_y', 'gyro_z']]
    acc = input.loc[:, ['acc_x', 'acc_y', 'acc_z']]
                            
    if gyro_max == None:
        gyro_max = np.abs(gyro.to_numpy()).max()
    gyro_min = -gyro_max

    if acc_max == None:
        acc_max = np.abs(acc.to_numpy()).max()
    acc_min = -acc_max

    input.loc[:, ['gyro_x', 'gyro_y', 'gyro_z']] = 2*(gyro - gyro_min) / (gyro_max - gyro_min) - 1
    input.loc[:, ['acc_x', 'acc_y', 'acc_z']] = 2*(acc - acc_min) / (acc_max - acc_min) - 1
    return input

# def minmax(input, gyro_max=4.2, acc_max=16.8):
#     gyro = input.loc[:, ['gyro_x', 'gyro_y', 'gyro_z']]
#     acc_x = input.loc[:, ['acc_x']]
#     acc_y = input.loc[:, ['acc_y']]
#     acc_z = input.loc[:, ['acc_z']]
                            
#     if gyro_max == None:
#         gyro_max = np.abs(gyro.to_numpy()).max()
#     gyro_min = -gyro_max

#     # if acc_max == None:
#     #     acc_max = np.abs(acc.to_numpy()).max()
#     # acc_min = -acc_max
#     acc_xy_max = 2
#     acc_xy_min = -acc_xy_max
#     acc_z_max = 12
#     acc_z_min = 8

#     input.loc[:, ['gyro_x', 'gyro_y', 'gyro_z']] = 2*(gyro - gyro_min) / (gyro_max - gyro_min) - 1
#     input.loc[:, ['acc_x']] = 2*(acc_x - acc_xy_min) / (acc_xy_max - acc_xy_min) - 1
#     input.loc[:, ['acc_y']] = 2*(acc_y - acc_xy_min) / (acc_xy_max - acc_xy_min) - 1
#     input.loc[:, ['acc_z']] = 2*(acc_z - acc_z_min) / (acc_z_max - acc_z_min) - 1
#     return input

def standardize(input, data_mean=None, data_std=None):
    if data_mean == None:
        data_mean = input.mean() 
    if data_std == None:
        data_std = input.std()
    input = (input - data_mean) / data_std
    return input

def uniformize(input,nbins=500):
    '''
    Creates a uniform distribution between 0 and 1 of a dataset according to a distribution. 
    '''
    output = input.copy()

    for col in ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']:
        x = list(output[col])
        which = lambda lst:list(np.where(lst)[0])

        gh = np.histogram(x,bins=nbins)
        empirical_cumulative_distribution = np.cumsum(gh[0])/nbins

        ans_x = x.copy()
        for idx in range(len(x)):
            max_idx = max(which(gh[1]<x[idx])+[0])
            ans_x[idx] = empirical_cumulative_distribution[max_idx] 
        output[col] = ans_x
    return output
