import logging

import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

import file
from convolution.database_sequence import DatabaseSequence
from src import config


def postprocess_features(features):
    return np.reshape(features, (features.shape[0], 7 * 7 * 512))


def transform_dataset(img_db_path, transformation_model):
    logging.debug("Load dataset {path}".format(path=img_db_path))

    total_cls_cnt = file.get_total_cls_cnt(config.set_path)
    assert total_cls_cnt != 0

    data_seq = DatabaseSequence(img_db_path, config.batch_size, total_cls_cnt, False)
    features = transformation_model.predict_generator(data_seq)
    labels = data_seq.labels
    data_seq.close()

    features_array = postprocess_features(features)

    return features_array, labels


def init_svm(gamma):
    # Default values
    # kernel = rbf (exponential kernel)
    # gamma = 1/n_features
    # verbose = False
    svm = SVC(kernel="rbf", gamma=gamma)
    return svm


def show_model(model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name)


def main():
    logging.basicConfig(filename="svm_classifier.log", level=logging.DEBUG)

    base_model = load_model(config.base_model_path)

    features_train, labels_train = transform_dataset(config.get_convolution_datasets_path('training'), base_model)
    features_test, labels_test = transform_dataset(config.get_convolution_datasets_path('test'), base_model)

    for gamma in config.svm_gamma_list:
        logging.info("Run parameters: gamma:{gamma}".format(gamma=gamma))
        # train
        svm = init_svm(gamma)
        svm.fit(features_train, labels_train)

        # validate
        y_predict = svm.predict(features_test)

        # save params
        cfmx = confusion_matrix(labels_test, y_predict)
        report = classification_report(labels_test, y_predict)

        logging.info("Confusion matrix:\n")
        logging.info(cfmx)
        logging.info("Classification report:\n")
        logging.info(report)


if __name__ == "__main__":
    main()
