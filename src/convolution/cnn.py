import logging
import math

import keras
from keras import Input, Model, callbacks
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

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


def build_cnn(layers, activation_fun, bottleneck_layers, dropout=False):
    logging.info("Building CNN "
                 "layers:{layers}, activation:{activation}, "
                 "bottleneck_layers: {bottleneck}, dropout: {dropout}"
                 .format(layers=layers,
                         activation=activation_fun,
                         bottleneck=bottleneck_layers,
                         dropout=dropout))

    inputs = Input(shape=(224, 224, 3))
    x = inputs

    channels = 32
    for level in range(layers):
        # Convolutions
        x = Conv2D(channels, (3, 3), activation=activation_fun, padding='same', name=str(level) + "_conv")(x)
        # number of filters = number of feature maps at the output

        # MaxPooling
        # 2*2 still keeps important features in-place the size of feature maps is deduced)
        # vector contains some spatial structure or some pixel patterns in the huge vector
        # size of the convolution layer divided by 2
        # size of the convolution layer divided by 2
        x = MaxPooling2D((2, 2), strides=(2, 2), name=str(level) + "_pool")(x)
        channels *= 2
        if channels > 1024:
            channels = 1024

    # Flattening
    x = GlobalAveragePooling2D()(x)
    # Flattening: takes all pooled feature maps and puts them into 1 single vector (HUGE!)
    # We keep spatial structure information in one huge vector that goes to ANN
    # each feature map -> one specific feature of an image

    # Full Connection (hidden)
    # number of hidden nodes between the number of output nodes and input nodes (changeable)
    # no rule of thumb for the best size - should be tested
    x = Dense(bottleneck_layers, activation='relu', name='dense')(x)
    total_cls_cnt = file.get_total_cls_cnt(config.set_path)

    # Dropout to test if reduces overfitting
    if dropout:
        x = Dropout(config.dropout_prob)(x)

    # Output Layer with the size = number of possible classes
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

    for bottleneck_layers in config.bottleneck_layer_sizes:
        for layer_cnt in range(config.layer_cnt_start, config.layer_cnt_stop + 1, config.layer_cnt_step):
            for activation in config.activation_functions:
                for loss_function in config.loss_functions:
                    model = build_cnn(layer_cnt, activation, bottleneck_layers, config.add_dropout)
                    train_model(model, gen_train, gen_validation, loss_function)
                    evaluate_model(model, gen_test)

                    if config.save_cnn_model:
                        logging.info("Saving base_model")
                        model.save(config.base_model_path)
                    keras.backend.clear_session()

    gen_test.close()
    gen_validation.close()
    gen_train.close()


if __name__ == "__main__":
    main()
