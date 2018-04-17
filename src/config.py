take_every_nth_sample = 4
clusters_count = 8

sizes_of_layers = [10, 10, 10]
training_fraction = 0.75
# test_fraction = 0.2
batch_size = 32
epochs = 6

resources_path = "../resources/"

set_path = resources_path + "SET_B/"
bounding_boxes_path = resources_path + "bounding_boxes.txt"


features_db_path = resources_path + "extracted_features_" + str(take_every_nth_sample) + ".hdf5"
clusters_db_path = resources_path + "clusters_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".hdf5"
labels_db_path = resources_path + "labels_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".hdf5"
