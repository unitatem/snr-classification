import logging

import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

import file
from convolution.database_sequence import DatabaseSequence
from src import config


def reshape_features(features):
    return np.reshape(features, (features.shape[0], 7 * 7 * 512))


def get_dataset(img_db_path, transformation_model):
    total_cls_cnt = file.get_total_cls_cnt(config.set_path)
    assert total_cls_cnt != 0

    data_seq = DatabaseSequence(img_db_path, config.batch_size, total_cls_cnt, False)
    features = transformation_model.predict_generator(data_seq)
    labels = data_seq.labels
    data_seq.close()

    features_array = reshape_features(features)

    return features_array, labels


def main():
    logging.basicConfig(filename="svm_classifier.log", level=logging.DEBUG)

    base_model = load_model(config.base_model_path)
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)

    # 'training'
    features_train, labels_train = get_dataset(config.get_convolution_datasets_path('validation'), base_model)

    svm = SVC()
    svm.fit(features_train, labels_train)

    features_test, labels_test = get_dataset(config.get_convolution_datasets_path('test'), base_model)
    y_predict = svm.predict(features_test)

    cfmx = confusion_matrix(labels_test, y_predict)
    print("Confusion matrix:\n")
    print(cfmx)
    report = classification_report(labels_test, y_predict)
    print("Classification report:\n")
    print(report)


if __name__ == "__main__":
    main()
