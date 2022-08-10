from torch import quantization
from datamodule.normalization import uniformize
from os import path, walk
import os

import numpy as np
import pandas as pd
from bagpy import bagreader
import matplotlib.pyplot as plt

from utils.quaternions import to_euler_angles_numpy, from_euler_angles_numpy

from datamodule.data_merging_gui import MergeDatasetsGui

def process_bagfile(filename, verbose=False):
    # Transform a single raw simulation .bag-file dataset and create corresponding .csv files
    # Load in a raw bagfile, create the corresponding .csv files
    if not path.exists(filename[:-4] + '/mu.csv'):
        if verbose:
            print(f"Creating .csv file for {filename}")
        bag = bagreader(filename)
        print(bag.topic_table)
        imu = bag.message_by_topic('imu')
        imu_gt = bag.message_by_topic('ground_truth/imu')
    else:
        if verbose: 
            print("this file has already been processed, skipping to save time")
        pass

def process_bagfile_EuRoC(filename, verbose=False):
    # Transform a single raw EuRoC .bag-file dataset and create corresponding .csv files
    # Load in a raw bagfile, create the corresponding .csv files
    if not path.exists(filename[:-4] + '/mu.csv'):
        if verbose:
            print(f"Creating .csv file for {filename}")
        bag = bagreader(filename)
        print(bag.topic_table)
        imu = bag.message_by_topic('/imu0')
        imu1 = bag.message_by_topic('/fcu/imu')
        pos = bag.message_by_topic('/vicon/firefly_sbx/firefly_sbx')
    else:
        if verbose: 
            print("this file has already been processed, skipping to save time")
        pass


def convert_dataset(filename, values):
    time, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, quat_w, quat_x, quat_y, quat_z, roll, pitch, yaw, acc_x_gt, acc_y_gt, acc_z_gt = values

    df = pd.DataFrame()

    df['Time'] = time
    df['gyro_x'], df['gyro_y'], df['gyro_z'] = gyro_x, gyro_y, gyro_z
    df['acc_x'], df['acc_y'], df['acc_z'] = acc_x, acc_y, acc_z
    df['quat_w'], df['quat_x'], df['quat_y'], df['quat_z'] = quat_w, quat_x, quat_y, quat_z
    df['roll'], df['pitch'], df['yaw'] = roll, pitch, yaw
    df['acc_x_gt'], df['acc_y_gt'], df['acc_z_gt'] = acc_x_gt, acc_y_gt, acc_z_gt
    
    # df['Time'] = pd.to_datetime(df['Time'] * 1000000000)
    # df.set_index('Time', inplace=True)
    # df = df.resample("10L").ffill().dropna()
    # df.reset_index(inplace=True)
    # print(df)
    df.to_csv(filename, index=False)


def convert_simulation_dataset(imu, odometry, filename_out):
    # Transform raw simulation .csv files into single dataset that can be directly used to create training/evaluation datasets
    df_imu = pd.read_csv(imu)
    df_odometry = pd.read_csv(odometry)

    quats = df_odometry[['pose.pose.orientation.w', 
                            'pose.pose.orientation.x', 
                            'pose.pose.orientation.y', 
                            'pose.pose.orientation.z']].to_numpy()
    eulers = to_euler_angles_numpy(quats)

    values = [df_imu['Time'], 
                df_imu['angular_velocity.x'],
                df_imu['angular_velocity.y'],
                df_imu['angular_velocity.z'],
                df_imu['linear_acceleration.x'],
                df_imu['linear_acceleration.y'],
                df_imu['linear_acceleration.z'],
                df_odometry['pose.pose.orientation.w'], # gt quaternions
                df_odometry['pose.pose.orientation.x'],
                df_odometry['pose.pose.orientation.y'],
                df_odometry['pose.pose.orientation.z'],
                eulers[:, 0],
                eulers[:, 1],
                eulers[:, 2],
                df_odometry['twist.twist.angular.x'], 
                df_odometry['twist.twist.angular.y'], 
                df_odometry['twist.twist.angular.z']]


    convert_dataset(filename_out, values)


def convert_crazyflie_dataset(filename, filename_out):
    # Transform a single crazyflie dataset .csv-file and create corresponding .csv files in same structure as simulation data. 
    df_imu = pd.read_csv(filename, skipinitialspace=True)

    eulers = df_imu.loc[:, ['roll', 'pitch', 'yaw']].to_numpy()
    quats = from_euler_angles_numpy(eulers)
    
    values = [df_imu['timeTick'] / 1000., 
                df_imu['gyroX'] / 180 * np.pi,
                df_imu['gyroY'] / 180 * np.pi,
                df_imu['gyroZ'] / 180 * np.pi,
                df_imu['accX'] * 9.81,
                df_imu['accY'] * 9.81,
                df_imu['accZ'] * 9.81,
                quats[:, 0],
                quats[:, 1],
                quats[:, 2],
                quats[:, 3],
                df_imu['roll'] / 180 * np.pi,
                -df_imu['pitch'] / 180 * np.pi, 
                df_imu['yaw'] / 180 * np.pi,
                0, #no ground truth, could be added
                0,
                0]

    convert_dataset(filename_out, values)

def convert_px4_dataset(filename, filename_out):
    # Transform px4 topic .csv-files for a single test-run and create corresponding .csv files in same structure as simulation data. 
    df_imu = pd.read_csv(f'{filename}_sensor_combined_0.csv', skipinitialspace=True)
    df_imu['Time'] = pd.to_datetime(df_imu['timestamp'], unit='us')
    df_imu.rename(columns={"gyro_rad[0]":"gyro_x",
                                "gyro_rad[1]":"gyro_y",
                                "gyro_rad[2]":"gyro_z",
                                "accelerometer_m_s2[0]":"acc_x",
                                "accelerometer_m_s2[1]":"acc_y",
                                "accelerometer_m_s2[2]":"acc_z"}, inplace=True)
    df_kalman = pd.read_csv(f'{filename}_estimator_states_0.csv', skipinitialspace=True)
    df_kalman['Time'] = pd.to_datetime(df_kalman['timestamp'], unit='us')
    df_kalman.rename(columns={"states[0]":"quat_x_kalman", 
                                "states[1]":"quat_y_kalman", 
                                "states[2]":"quat_z_kalman", 
                                "states[3]":"quat_w_kalman"}, inplace=True)
    # calculate euler angles from kalman quaternion
    eulers = to_euler_angles_numpy(df_kalman.loc[:, ['quat_w_kalman', 'quat_x_kalman', 'quat_y_kalman', 'quat_z_kalman']].to_numpy())
    df_kalman['pitch_kalman'], df_kalman['roll_kalman'], df_kalman['yaw_kalman'] = eulers[:, 1], eulers[:, 0], eulers[:, 2]

    df_px4 = pd.merge_asof(df_imu, df_kalman, direction='nearest', on='Time')
    df_px4.set_index('Time', inplace=True)
    df_px4 = df_px4[~df_px4.index.duplicated()].resample("4989U").ffill().dropna() #4989 is microseconds from px4 data
    df_px4.reset_index(inplace=True)

    df_opti = pd.read_csv(f'{filename}_opti.csv', 
                        skipinitialspace=True, 
                        skiprows=6,
                        usecols=range(9), 
                        header=0, 
                        names=['Frame', 'Time', 'quat_x', 'quat_y', 'quat_z', 'quat_w', 'x', 'y', 'z'])

    # Create time column to match the time in PX4 microseconds (and start almost at the same point)
    df_opti['timestamp'] = df_opti['Time'] * 1e6  + df_imu.timestamp[0] # change to microseconds to match px4 data
    df_opti['Time'] = pd.to_datetime(df_opti.timestamp, unit='us')
    df_opti.set_index('Time', inplace=True)

    # Upsample and interpolate optitrack data to match PX4 frequency
    df_opti = df_opti.resample("4989U").ffill(limit=1).interpolate(method='linear').dropna()
    df_opti.reset_index(inplace=True)
    
    # calculate euler angles from optitrack quaternion
    eulers = to_euler_angles_numpy(df_opti.loc[:, ['quat_w', 'quat_x', 'quat_y', 'quat_z']].to_numpy())
    df_opti['pitch'], df_opti['roll'], df_opti['yaw'] = -eulers[:, 0], eulers[:, 1], -eulers[:, 2]


    df_merged = merge_px4_data(df_px4, df_opti)
    
    # NOTE: SIGNS BEFORE SENSOR MEASUREMENTS ARE TO ACCOUNT FOR REFERENCE FRAME DIFFERENCES
    values = [df_merged['Time'], 
                df_merged['gyro_x'],
                -df_merged['gyro_y'],
                -df_merged['gyro_z'],
                df_merged['acc_x'],
                -df_merged['acc_y'],
                -df_merged['acc_z'], 
                df_merged['quat_x'], #FROM OPTI
                df_merged['quat_y'],
                df_merged['quat_z'],
                df_merged['quat_w'],
                df_merged['yaw'], # ROLL IS YAW IN PX4 DATA NOW
                df_merged['pitch'], 
                df_merged['roll'],
                0, #no ground truth, could be added
                0,
                0]

    convert_dataset(filename_out, values)

def merge_px4_data(df_px4, df_opti):
    '''Combine logged px4 data and measured ground truth from optitrack and combine into single dataset'''

    # Open GUI to let user decide the offset
    app = MergeDatasetsGui(df_px4, df_opti)
    app.mainloop()
    offset = app.start_time_offset

    # Change timestamp of optitrack data to match this offset and merge with px4 data
    # df_opti.timestamp = df_opti.timestamp + np.sign(offset) * (df_opti.timestamp[np.abs(int(offset))] - df_opti.timestamp[0])
    # df_opti.Time = pd.to_datetime(df_opti.timestamp, unit='us')
    df_opti.Time = df_opti.Time + offset
    df_merged = pd.merge_asof(df_px4, 
                                df_opti, 
                                on='Time', 
                                direction='nearest',
                                tolerance=pd.Timedelta("5000U")).dropna()
    return df_merged   

def convert_EuRoC_dataset(imu, vicon, filename_out):
        # Transform raw simulation .csv files into single dataset that can be directly used to create training/evaluation datasets
    df_imu = pd.read_csv(imu)
    df_vicon = pd.read_csv(vicon)


    df_imu['timestamp'] = df_imu['Time'] * 1e6  # change to microseconds to match px4 data
    df_imu['Time'] = pd.to_datetime(df_imu.timestamp, unit='us')
    df_imu.set_index('Time', inplace=True)
    df_imu = df_imu[~df_imu.index.duplicated()].resample("4989U").ffill().dropna() #4989 is microseconds from px4 data
    df_imu.reset_index(inplace=True)

    df_vicon['timestamp'] = df_vicon['Time'] * 1e6  # change to microseconds to match px4 data
    df_vicon['Time'] = pd.to_datetime(df_vicon.timestamp, unit='us')
    df_vicon.set_index('Time', inplace=True)
    df_vicon = df_vicon[~df_vicon.index.duplicated()].resample("4989U").ffill().dropna()
    df_vicon.reset_index(inplace=True)

    df_merged = pd.merge_asof(df_imu, 
                                df_vicon, 
                                on='Time', 
                                direction='nearest',
                                tolerance=pd.Timedelta("5000U")).dropna()

    quats = df_merged[['transform.rotation.w', 'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z']].to_numpy()
    eulers = to_euler_angles_numpy(quats)

    values = [df_merged['Time'], 
                df_merged['angular_velocity.x'],
                df_merged['angular_velocity.y'],
                df_merged['angular_velocity.z'],
                df_merged['linear_acceleration.x'],
                df_merged['linear_acceleration.y'],
                df_merged['linear_acceleration.z'],
                df_merged['transform.rotation.w'],
                df_merged['transform.rotation.x'],
                df_merged['transform.rotation.y'],
                df_merged['transform.rotation.z'],
                eulers[:, 0],
                eulers[:, 1],
                eulers[:, 2],
                0,
                0,
                0]

    convert_dataset(filename_out, values)


def create_training_dataset(data, options):
    encoding = options['encoding']
    normalization = options['normalization']
    skipinitial = options['skipinitial']
    nbins = options['nbins']

    if skipinitial != None:
        data = data.iloc[skipinitial:]
        data.reset_index(inplace=True)

    if options['add_sensor_noise']:
        data['gyro_x'] = data['gyro_x'] + np.random.randn(len(data['gyro_x'])) * 0.03
        data['gyro_y'] = data['gyro_y'] + np.random.randn(len(data['gyro_y'])) * 0.03
        data['gyro_z'] = data['gyro_z'] + np.random.randn(len(data['gyro_z'])) * 0.03
        data['acc_x'] = data['acc_x'] + np.random.randn(1) * 0.01
        data['acc_y'] = data['acc_y'] + np.random.randn(1) * 0.01
        data['acc_z'] = data['acc_z'] + np.random.randn(1) * 0.01
        
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
    if options['data_type'] in ['simulation', 'EuRoC']:
        for (directory, dirnames, filenames) in walk(input_folder, topdown=True):
            for dirname in dirnames:
                for (directory, dirnames, filenames) in walk(path.join(input_folder, dirname), topdown=True):
                    for filename in filenames:
                        if path.splitext(filename)[1] == '.bag':
                            if options['data_type'] == 'simulation':
                                process_bagfile(path.join(directory, filename), verbose=True)
                            elif options['data_type'] == 'EuRoC':
                                process_bagfile_EuRoC(path.join(directory, filename), verbose=True)
            break
    for (directory, dirnames, filenames) in walk(input_folder, topdown=True):
        for dirname in dirnames:
            filename_out = path.join(input_folder, f'{dirname}/{dirname}_converted.csv')
            if options['data_type'] == 'simulation':
                imu_filename = path.join(input_folder, f'{dirname}/{dirname}/imu.csv')
                odometry_filename = path.join(input_folder, f'{dirname}/{dirname}/odometry.csv')
                
                if not path.exists(filename_out):
                    convert_simulation_dataset(imu_filename, odometry_filename, filename_out)
            elif options['data_type'] == 'crazyflie':
                filename = path.join(input_folder, f'{dirname}/{dirname}.csv')
                if not path.exists(filename_out):
                    convert_crazyflie_dataset(filename, filename_out)
            elif options['data_type'] == 'px4':
                filename = path.join(input_folder, f'{dirname}/{dirname}')
                if not path.exists(filename_out):
                    convert_px4_dataset(filename, filename_out)
            elif options['data_type'] == 'EuRoC':
                imu_filename = path.join(input_folder, f'{dirname}/{dirname}/fcu-imu.csv')
                pos_filename = path.join(input_folder, f'{dirname}/{dirname}/vicon-firefly_sbx-firefly_sbx.csv')
                if not path.exists(filename_out):
                    convert_EuRoC_dataset(imu_filename, pos_filename, filename_out)
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