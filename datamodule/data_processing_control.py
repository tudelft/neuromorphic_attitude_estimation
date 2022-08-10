from datamodule.normalization import uniformize
from os import path, walk
import os

import numpy as np
import pandas as pd
from bagpy import bagreader

from utils.quaternions import to_euler_angles_numpy, from_euler_angles_numpy
from datamodule.body_equations import calculate_allocation_matrix, omega_to_forces

def process_bagfile(filename, verbose=False):
    # Transform a single raw simulation .bag-file dataset and create corresponding .csv files
    # Load in a raw bagfile, create the corresponding .csv files
    if not path.exists(filename[:-4] + '/imu.csv'):
        if verbose:
            print(f"Creating .csv files for {filename}")
        bag = bagreader(filename)
        print(bag.topic_table)
        imu = bag.message_by_topic('/imu')
        odometry = bag.message_by_topic('/odometry')
        odometry_estimate = bag.message_by_topic('/odometry_estimate')
        motor_speed = bag.message_by_topic('/command/motor_speed')
    else:
        if verbose: 
            print("this file has already been processed, skipping to save time")
        pass


def convert_dataset(filename, values, frequency):
    time_imu, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, \
        time_odo, quat_w, quat_x, quat_y, quat_z, roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel, \
        time_motors, thrust, torque_pitch, torque_roll, torque_yaw = values

    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    df_3 = pd.DataFrame()

    df_1['Time'] = time_imu
    df_1['gyro_x'], df_1['gyro_y'], df_1['gyro_z'] = gyro_x, gyro_y, gyro_z
    df_1['acc_x'], df_1['acc_y'], df_1['acc_z'] = acc_x, acc_y, acc_z
    df_2['Time'] = time_odo
    df_2['quat_w'], df_2['quat_x'], df_2['quat_y'], df_2['quat_z'] = quat_w, quat_x, quat_y, quat_z
    df_2['roll'], df_2['pitch'], df_2['yaw'] = roll, pitch, yaw
    df_2['roll_vel'], df_2['pitch_vel'], df_2['yaw_vel'] = roll_vel, pitch_vel, yaw_vel
    df_3['Time'] = time_motors
    # df_3['thrust'], df_3['torque_pitch'], df_3['torque_roll'], df_3['torque_yaw'] = thrust, torque_pitch, torque_roll, torque_yaw


    df = df_1.merge(df_2, on="Time", how="outer")
    df = df.merge(df_3, how='outer', on='Time')
    df['Time'] = pd.to_datetime(df['Time'] * 1000000000)
    df.set_index('Time', inplace=True)

    # Check if this dict fixes the hardcoded issue. Necessary to add other frequencies if they are not in here.
    resampling_freq_dict = dict({'100hz':"10L", '200hz':"5L", '500hz':"2L"})
    df = df[~df.index.duplicated()].resample(resampling_freq_dict[str(frequency)]).ffill().dropna()
    df.reset_index(inplace=True)
    df.to_csv(filename, index=False)


def convert_simulation_dataset(imu, odometry, motor_speed, filename_out, frequency):
    # Transform raw simulation .csv files into single dataset that can be directly used to create training/evaluation datasets
    df_imu = pd.read_csv(imu)
    df_odometry = pd.read_csv(odometry)
    df_motor_speed = pd.read_csv(motor_speed)

    # print(df_odometry)
    # create ground truth euler angles
    quats = df_odometry[['pose.pose.orientation.w', 
                            'pose.pose.orientation.x', 
                            'pose.pose.orientation.y', 
                            'pose.pose.orientation.z']].to_numpy()

    eulers = to_euler_angles_numpy(quats)

    # transform rotor speeds to acceleration and torques
    K_F = 8.54858e-06
    L = 0.17
    K_D = K_F * 0.016
    mw = calculate_allocation_matrix(K_F, K_D, L, '+')
    omega = df_motor_speed[['angular_velocities_0', 
                            'angular_velocities_1', 
                            'angular_velocities_2', 
                            'angular_velocities_3']].to_numpy().T
    T, tau = omega_to_forces(omega, mw)
    # print(T, tau)
    
    values = [df_imu['Time'], 
                df_imu['angular_velocity.x'],           # gyroscope
                df_imu['angular_velocity.y'],
                df_imu['angular_velocity.z'],
                df_imu['linear_acceleration.x'],        # accelerometer
                df_imu['linear_acceleration.y'],
                df_imu['linear_acceleration.z'],
                df_odometry['Time'],
                df_odometry['pose.pose.orientation.w'], # gt quaternions
                df_odometry['pose.pose.orientation.x'],
                df_odometry['pose.pose.orientation.y'],
                df_odometry['pose.pose.orientation.z'],
                eulers[:, 0],                           # gt eulers
                eulers[:, 1],
                eulers[:, 2],
                df_odometry['twist.twist.angular.x'], 
                df_odometry['twist.twist.angular.y'], 
                df_odometry['twist.twist.angular.z'], 
                df_motor_speed['Time'],
                T,                                      # thrust and torque commands
                tau[0],
                tau[1], 
                tau[2]
                ]
    convert_dataset(filename_out, values, frequency)


# def convert_crazyflie_dataset(filename, filename_out, frequency):
#     # Transform a single crazyflie dataset .csv-file and create corresponding .csv files in same structure as simulation data. 
#     df_imu = pd.read_csv(filename, skipinitialspace=True)

#     eulers = df_imu.loc[:, ['roll', 'pitch', 'yaw']].to_numpy()
#     quats = from_euler_angles_numpy(eulers)
    
#     values = [df_imu['timeTick'] / 1000., 
#                 df_imu['gyroX'] / 180 * np.pi,
#                 df_imu['gyroY'] / 180 * np.pi,
#                 df_imu['gyroZ'] / 180 * np.pi,
#                 df_imu['accX'] * 9.81,
#                 df_imu['accY'] * 9.81,
#                 df_imu['accZ'] * 9.81,
#                 quats[:, 0],
#                 quats[:, 1],
#                 quats[:, 2],
#                 quats[:, 3],
#                 df_imu['roll'] / 180 * np.pi,
#                 -df_imu['pitch'] / 180 * np.pi, 
#                 df_imu['yaw'] / 180 * np.pi,
#                 0, #no ground truth, could be added
#                 0,
#                 0]

#     convert_dataset(filename_out, values, frequency)


def create_training_dataset(data, options):
    encoding = options['encoding']
    normalization = options['normalization']
    skipinitial = options['skipinitial']
    nbins = options['nbins']

    if skipinitial != None:
        data = data.iloc[skipinitial:]
        data.reset_index(inplace=True)

    imu_data = data.loc[:, ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']]
    
    if normalization != None:
        # Currently output is not considered for normalization. Could be interesting
        if normalization == uniformize:
            imu_data = normalization(imu_data, nbins=len(imu_data))
        else:
            imu_data = normalization(imu_data)
        data.loc[:, ['gyro_x_norm', 'gyro_y_norm', 'gyro_z_norm', 'acc_x_norm', 'acc_y_norm', 'acc_z_norm']] = imu_data.to_numpy()
    else:
        data.loc[:, ['gyro_x_norm', 'gyro_y_norm', 'gyro_z_norm', 'acc_x_norm', 'acc_y_norm', 'acc_z_norm']] = 0

    if encoding != None:
        if normalization == uniformize:
            min_val = 0
            max_val = 1
        else:
            min_val = -1
            max_val = 1

        encoded_imu_data = encoding(imu_data, nbins, min_val, max_val).squeeze(0)
        for i in range(encoded_imu_data.size()[1]):
            data[f'n{i}'] = encoded_imu_data[:, i]
    return data


def create_standard_form_files_from_folder(input_folder, options):
    if options['data_type'] == 'simulation':
        for (directory, dirnames, filenames) in walk(input_folder, topdown=True):
            for dirname in dirnames:
                for (directory, dirnames, filenames) in walk(path.join(input_folder, dirname), topdown=True):
                    for filename in filenames:
                        if path.splitext(filename)[1] == '.bag':
                            process_bagfile(path.join(directory, filename), verbose=True)
            break
    for (directory, dirnames, filenames) in walk(input_folder, topdown=True):
        for dirname in dirnames:
            filename_out = path.join(input_folder, f'{dirname}/{dirname}_converted.csv')
            if options['data_type'] == 'simulation':
                imu_filename = path.join(input_folder, f'{dirname}/{dirname}/imu.csv')
                odometry_filename = path.join(input_folder, f'{dirname}/{dirname}/odometry.csv')
                motor_speed_filename = path.join(input_folder, f'{dirname}/{dirname}/command-motor_speed.csv')
                
                if not path.exists(filename_out):
                    convert_simulation_dataset(imu_filename, odometry_filename, motor_speed_filename, filename_out, options["freq"])
            # elif options['data_type'] == 'crazyflie':
            #     filename = path.join(input_folder, f'{dirname}/{dirname}.csv')
            #     if not path.exists(filename_out):
            #         convert_crazyflie_dataset(filename, filename_out)
        break
    

def create_training_dataset_folder(input_folder, training_dataset_folder, options):
    output_filename = f"{options['data_type']}_{options['freq']}_{options['encoding_name']}_{options['nbins']}bins_{options['normalization_name']}"
    output_folder = path.join(training_dataset_folder, output_filename)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(path.join(output_folder, 'Train'))
        os.mkdir(path.join(output_folder, 'Validation'))
        
    for (directory, dirnames, filenames) in walk(input_folder, topdown=True):
        number_of_files = len(dirnames)
        number_of_train_files = np.floor(0.9 * number_of_files) #currently hardcoded split of 90% train and 10% validation data
        print(number_of_train_files)
        number_of_validation_files = number_of_files - number_of_train_files

        split_counter = 0
        for dirname in dirnames:
            split_name = "Train" if split_counter < number_of_train_files else "Validation"
            split_counter += 1
            
            filename = path.join(input_folder, f'{dirname}/{dirname}_converted.csv')
            data = pd.read_csv(filename)

            training_set = create_training_dataset(data, options)
            print(output_folder)
            out_filename = path.join(output_folder, 
                                     f"{split_name}/{dirname}_{output_filename}.csv")
            print(out_filename)
            training_set.to_csv(out_filename)
        break