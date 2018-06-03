import os

take_every_nth_sample = 1
data_multiplication_factor = 1

# every range in config is both side closed <a, b>
clusters_count_start = 8
clusters_count_stop = 20
clusters_count_step = 12


activation_functions = ['sigmoid', 'tanh', 'relu', 'elu']

layer_cnt_start = 1
layer_cnt_stop = 4
layer_cnt_step = 1

layer_size_start = 50
layer_size_stop = 100
layer_size_step = 50

filter_channels_start = 8
filter_channels_stop = 64
filter_channels_ratio = 2

# ref: https://keras.io/losses/
loss_functions = ["mean_squared_error", "mean_squared_logarithmic_error",
                  "logcosh", "categorical_crossentropy", "poisson"]

save_cnn_model = False

bottleneck_layer_sizes = [128, 256, 512]

add_dropout = 0


svm_gamma_list = ['auto', 1, 2, 3, 5, 8, 13, 21, 34]

training_total_ratio = 0.5
validation_total_ratio = 0.3

batch_size = 4  # 64
max_epochs = 20  # 200
min_improvement_required = 0.001
max_no_improvement_epochs = 2

resources_path = "../../resources"

set_path = os.path.join(resources_path, "SET_B")
bounding_boxes_path = os.path.join(resources_path, "bounding_boxes.txt")


def get_custom_extension_without_multiplication():
    return str(take_every_nth_sample) + '.hdf5'


def get_custom_extension():
    return str(take_every_nth_sample) + 'm' + str(data_multiplication_factor) + '.hdf5'


features_db_path = os.path.join(resources_path, "extracted_features_" + get_custom_extension())

data_type = ['training', 'validation', 'test']

groups_db_path = {'training': os.path.join(resources_path, "training_features_" + get_custom_extension()),
                  'validation': os.path.join(resources_path, "validation_features_" + get_custom_extension()),
                  'test': os.path.join(resources_path, "test_features_" + get_custom_extension())}


def get_clusters_db_path(data_type, clusters_count):
    transformed_groups_db_path = {
        'training': os.path.join(resources_path, "training_clusters_" + str(clusters_count) + '_' + get_custom_extension()),
        'validation': os.path.join(resources_path, "validation_clusters_" + str(clusters_count) + '_' + get_custom_extension()),
        'test': os.path.join(resources_path, "test_clusters_" + str(clusters_count) + '_' + get_custom_extension())}
    return transformed_groups_db_path[data_type]


def get_labels_db_path(clusters_count):
    return os.path.join(resources_path, "labels_" + str(clusters_count) + '_' + get_custom_extension())


def get_convolution_datasets_path(key):
    return os.path.join(resources_path, 'dataset_' + key + '_' + get_custom_extension_without_multiplication())


base_model_path = os.path.join(resources_path, "base_model.hdf5")
