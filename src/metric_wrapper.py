from keras import metrics


def top_1_accuracy(y_true, y_pred):
    """

    :param y_true: original labels
    :param y_pred: predicted labels
    :return: top 1 keras metrics
    """
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    """

    :param y_true: original labels
    :param y_pred: predicted labels
    :return: top 5 keras metrics
    """
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)