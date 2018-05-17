import os

take_every_nth_sample = 1
data_multiplication_factor = 3

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

training_total_ratio = 0.5
validation_total_ratio = 0.3

batch_size = 64
max_epochs = 200
min_improvement_required = 0.001
max_no_improvement_epochs = 2

resources_path = "../../resources"

set_path = os.path.join(resources_path, "SET_B")
bounding_boxes_path = os.path.join(resources_path, "bounding_boxes.txt")

features_db_path = os.path.join(resources_path,
                                "extracted_features_" + str(take_every_nth_sample) + "m" + str(
                                    data_multiplication_factor) + ".hdf5")

data_type = ['training', 'validation', 'test']

groups_db_path = {'training': os.path.join(resources_path,
                                           "training_features_" + str(take_every_nth_sample) + "m" + str(
                                               data_multiplication_factor) + ".hdf5"),
                  'validation': os.path.join(resources_path,
                                             "validation_features_" + str(take_every_nth_sample) + "m" + str(
                                                 data_multiplication_factor) + ".hdf5"),
                  'test': os.path.join(resources_path,
                                       "test_features_" + str(take_every_nth_sample) + "m" + str(
                                           data_multiplication_factor) + ".hdf5")}


def get_clusters_db_path(data_type, clusters_count):
    transformed_groups_db_path = {
        'training': os.path.join(resources_path,
                                 "training_clusters_" + str(take_every_nth_sample) + "m" + str(
                                     data_multiplication_factor) + "_" + str(clusters_count) + ".hdf5"),
        'validation': os.path.join(resources_path,
                                   "validation_clusters_" + str(take_every_nth_sample) + "m" + str(
                                       data_multiplication_factor) + "_" + str(clusters_count) + ".hdf5"),
        'test': os.path.join(resources_path,
                             "test_clusters_" + str(take_every_nth_sample) + "m" + str(
                                 data_multiplication_factor) + "_" + str(clusters_count) + ".hdf5")}
    return transformed_groups_db_path[data_type]


def get_labels_db_path(clusters_count):
    return os.path.join(resources_path,
                        "labels_" + str(take_every_nth_sample) + "m" + str(data_multiplication_factor) + "_" + str(
                            clusters_count) + ".hdf5")
