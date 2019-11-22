# general parameters
efficient_net_type = 'B0'
image_size = 224    # actual image size in pixels
train_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN_split'
test_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/VALIDATION'

'''
# parameters for ultimate layer training
num_of_dense_layers = 0
dense_layer_dim = 32
epochs_ul = 1
#steps_per_epoch_train_ul = 500
steps_per_epoch_train_ul = 5
steps_per_epoch_val_ul = 20
out_path = './model_ultimate_with_proper_validation_{}'.format(efficient_net_type)
'''
# parameters for fine tuning training
#epochs_ft = 40*8*4
epochs_ft = 4
#steps_per_epoch_train_ft = 500
steps_per_epoch_train_ft = 5
#steps_per_epoch_val_ft = 80
steps_per_epoch_val_ft = 8

out_path_ft = './model_ultimate_with_proper_validation_{}'.format(efficient_net_type)

# shifting for data augmentation, will be set to 0 in efficient_net_type == 'B0'
width_shift_range = 10
height_shift_range = 10
# determine input size from model name. input size is fixed
if efficient_net_type == 'B0':
    input_size = 224  # input size needed for network in pixels
    width_shift_range = 0
    height_shift_range = 0
    batch_size_ul = 512 # for p2
    batch_size_ft = 64 # for p2
elif efficient_net_type == 'B3':
    input_size = 300
elif efficient_net_type == 'B4':
    input_size = 380
    batch_size_ul = 32
    batch_size_ft = 8

lr = 10**(-3)
# interval size 0.4286, in paper: ration decay/lr ~ 10*(-6) to 10**(-3) at a batch size of 256
# the last term : batch_size /256 is only an approximation for the difference in batch size, since we do not have a linear decay
decay = 10**(-4.5) * lr *batch_size_ft/256.
momentum = 0.9
nesterov = True



lower_decay = 0.000001 * lr * batch_size_ft/256.
upper_decay = 0.001 * lr * batch_size_ft/256.

random_seed = 42
