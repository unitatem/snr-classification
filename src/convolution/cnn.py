import logging
import math

import keras
from keras import Input, Model, callbacks
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

import config
import file
import metric
from convolution.database_sequence import DatabaseSequence


def get_sequence_gen(img_db_path):
    logging.debug("Load dataset {path}".format(path=img_db_path))

    total_cls_cnt = file.get_total_cls_cnt(config.set_path)
    assert total_cls_cnt != 0

    data_seq = DatabaseSequence(img_db_path, config.batch_size, total_cls_cnt, True)
    return data_seq


def build_cnn(descriptor, layers, activation_fun, channels):
    logging.info("Building CNN "
                 "{{descriptor:{descriptor}, layers:{layers}, activation:{activation}, channels:{channels}}}"
                 .format(descriptor=descriptor,
                         layers=layers,
                         activation=activation_fun,
                         channels=channels))

    inputs = Input(shape=(224, 224, 3))

    x = inputs
    for level in range(layers):
        x = Conv2D(channels, (3, 3), activation=activation_fun, padding='same', name=str(level) + "_conv")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name=str(level) + "_pool")(x)
        channels *= 2
        if channels > 1024:
            channels = 1024

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='dense')(x)
    total_cls_cnt = file.get_total_cls_cnt(config.set_path)
    x = Dense(total_cls_cnt, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='ExperimentalModel')

    return model


def train_model(model, gen_train, gen_validation, loss_fun):
    logging.info("Train model: {{loss_fun:{loss}}}".format(loss=loss_fun))

    stop_callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required,
                                            patience=config.max_no_improvement_epochs)

    model.compile(optimizer='rmsprop',
                  loss=loss_fun,
                  metrics=[metric.top_1_accuracy,
                           metric.top_5_accuracy])
    model.fit_generator(gen_train,
                        validation_data=gen_validation,
                        epochs=config.max_epochs,
                        callbacks=[stop_callback])
    return model


def evaluate_model(model, gen_test):
    scores = model.evaluate_generator(gen_test)
    for i in range(len(scores)):
        logging.info("{name}: {value}".format(name=model.metrics_names[i],
                                              value=scores[i]))


def main():
    logging.basicConfig(filename="cnn.log", level=logging.DEBUG)

    gen_train = get_sequence_gen(config.get_convolution_datasets_path('training'))
    gen_validation = get_sequence_gen(config.get_convolution_datasets_path('validation'))
    gen_test = get_sequence_gen(config.get_convolution_datasets_path('test'))

    channels_range = [config.filter_channels_start * (2 ** m) for m in
                      range(0, int(math.log(config.filter_channels_stop / config.filter_channels_start, 2) + 1))]

    # descriptor?
    for layer_cnt in range(config.layer_cnt_start, config.layer_cnt_stop + 1, config.layer_cnt_step):
        for activation in config.activation_functions:
            for channels in channels_range:
                for loss_function in config.loss_functions:
                    model = build_cnn(1, layer_cnt, activation, channels)
                    model = train_model(model, gen_train, gen_validation, loss_function)

                    evaluate_model(model, gen_test)
                    keras.backend.clear_session()

    gen_test.close()
    gen_validation.close()
    gen_train.close()


if __name__ == "__main__":
    main()
