take_every_nth_sample = 4

# every range in config is both side closed <a, b>
clusters_count_start = 8
clusters_count_stop = 20
clusters_count_step = 12

# activation_functions = ['sigmoid', 'tanh', 'relu', 'elu']
activation_functions = ['sigmoid', 'tanh']

layer_cnt_start = 1
layer_cnt_stop = 4
layer_cnt_step = 4

layer_size_start = 50
layer_size_stop = 100
layer_size_step = 50

training_total_ratio = 0.75
batch_size = 64
max_epochs = 200
min_improvement_required = 0.001
max_no_improvement_epochs = 2

resources_path = "../resources/"

set_path = resources_path + "SET_B/"
bounding_boxes_path = resources_path + "bounding_boxes.txt"

features_db_path = resources_path + "extracted_features_" + str(take_every_nth_sample) + ".hdf5"


def get_clusters_db_path(clusters_count):
    return resources_path + "clusters_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".hdf5"


def get_labels_db_path(clusters_count):
    return resources_path + "labels_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".hdf5"
