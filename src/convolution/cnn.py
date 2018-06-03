import logging
import math

import keras
from keras import Input, Model, callbacks
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

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


def create_augmented_gens():
    """
    creates many batches of images; each batch = random transformation of images -> as a result we get many more
    diverse images to reduce overfitting
    :type model: cnn model
    :return:
    """
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,  # random transvections
        zoom_range=0.2,  # random zooms
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator()
    return train_datagen, validation_datagen


def build_cnn(descriptor, layers, activation_fun, channels, bottleneck_layers, dropout=False):
    logging.info("Building CNN "
                 "{{descriptor:{descriptor}, layers:{layers}, activation:{activation}, channels:{channels},"
                 " bottleneck_layers: {bottleneck}, dropout: {dropout}"
                 .format(descriptor=descriptor,
                         layers=layers,
                         activation=activation_fun,
                         channels=channels,
                         bottleneck=bottleneck_layers,
                         dropout=int(dropout)))

    inputs = Input(shape=(224, 224, 3))

    x = inputs
    for level in range(layers):
        #  1 - Convolutions
        x = Conv2D(channels, (3, 3), activation=activation_fun, padding='same', name=str(level) + "_conv")(x)
        # number of filters = number of feature maps at the output

        # 2 - MaxPooling
        # 2*2 still keeps important features in-place the size of feature maps is deduced)
        # vector contains some spatial structure or some pixel patterns in the huge vector
        # size of the convolution layer divided by 2

        x = MaxPooling2D((2, 2), strides=(2, 2), name=str(level) + "_pool")(x)
        if dropout and (level == range(layers)[-1]):
            x = Dropout(config.dropout_prob)(x)
        channels *= 2
        if channels > 1024:
            channels = 1024

    # 3 - Flattening
    x = GlobalAveragePooling2D()(x)
    # Flattening: takes all pooled feature maps and puts them into 1 single vector (HUGE!)
    # We keep spatial structure information in one huge vector that goes to ANN
    # each feature map -> one specific feature of an image

    #  4 - Full Connection (hidden)
    # number of hidden nodes between the number of output nodes and input nodes (changeable)
    # no rule of thumb for the best size - should be tested
    x = Dense(bottleneck_layers, activation='relu', name='dense')(x)
    total_cls_cnt = file.get_total_cls_cnt(config.set_path)

    #  5 - Output Layer with the size = number of possible classes
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

    train_datagen, validation_datagen = create_augmented_gens()

    model.fit_generator(train_datagen.flow(gen_train, batch_size=config.data_multiplication_factor),
                        validation_data=validation_datagen.flow(gen_validation,
                        batch_size=config.data_multiplication_factor),
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

    for bottleneck_layers in config.bottleneck_layer_sizes:
        for layer_cnt in range(config.layer_cnt_start, config.layer_cnt_stop + 1, config.layer_cnt_step):
            for activation in config.activation_functions:
                for channels in channels_range:
                    for loss_function in config.loss_functions:
                        model = build_cnn(1, layer_cnt, activation, channels, bottleneck_layers, bool(config.add_dropout))
                        model = train_model(model, gen_train, gen_validation, loss_function)
                        evaluate_model(model, gen_test)
                        keras.backend.clear_session()


    gen_test.close()
    gen_validation.close()
    gen_train.close()


if __name__ == "__main__":
    main()
