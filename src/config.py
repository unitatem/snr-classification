take_every_nth_sample = 1
clusters_count = 8

sizes_of_layers = [200, 200]
training_total_ratio = 0.5
validation_total_ratio = 0.3
batch_size = 64
max_epochs = 200
min_improvement_required = 0.001
max_no_improvement_epochs = 2

resources_path = "../resources/"

set_path = resources_path + "SET_B/"
bounding_boxes_path = resources_path + "bounding_boxes.txt"

groups_db_path = {'training': resources_path + "training_features_" + str(take_every_nth_sample) + ".hdf5",
                  'validation': resources_path + "validation_features_" + str(take_every_nth_sample) + ".hdf5",
                  'test': resources_path + "test_features_" + str(take_every_nth_sample) + ".hdf5"}
clusters_groups_db_path = {'training': resources_path + "training_clusters_" + str(take_every_nth_sample) + ".hdf5",
                           'validation': resources_path + "validation_clusters_" + str(take_every_nth_sample) + ".hdf5",
                           'test': resources_path + "test_clusters_" + str(take_every_nth_sample) + ".hdf5"}
features_db_path = resources_path + "extracted_features_" + str(take_every_nth_sample) + ".hdf5"
labels_db_path = resources_path + "labels_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".hdf5"
