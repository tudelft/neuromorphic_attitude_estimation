# from datamodule.data_processing_control import create_standard_form_files_from_folder, create_training_dataset_folder
from datamodule.data_processing import create_standard_form_files_from_folder, create_training_dataset_folder
from datamodule.encoding import grf_coding, stochastic_position_coding, position_coding
from datamodule.normalization import minmax, standardize, uniformize
# print(convert_simulation_dataset("/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/100hz/1_100hz/mu.csv", 
#                                  "/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/100hz/1_100hz/round_truth-imu.csv"))


options = {}
options['data_type'] = 'px4'
options['add_sensor_noise'] = False
options['encoding'] = None
options['encoding_name'] = ''
options['nbins'] = ''
options['normalization'] = minmax
options['normalization_name'] = 'minmax'
options['skipinitial'] = 100
options['freq'] = '200hz'

input_folder = "/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/revision_tests_copy"
output_folder = "/home/sstroobants/ownCloud/PhD/Code/IMU_NN/data/training_datasets_copy"
create_standard_form_files_from_folder(input_folder, options)
create_training_dataset_folder(input_folder, output_folder, options)