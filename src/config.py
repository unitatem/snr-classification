take_every_nth_sample = 4
clusters_count = 8

resources_path = "../resources/"

set_path = resources_path + "SET_B/"
bounding_boxes_path = resources_path + "bounding_boxes.txt"

features_db_path = resources_path + "extracted_features_" + str(take_every_nth_sample) + ".db"
clusters_db_path = resources_path + "clusters_" + str(take_every_nth_sample) + "_" + str(clusters_count) + ".db"
