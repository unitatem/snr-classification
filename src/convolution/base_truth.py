import logging

from keras import Model, callbacks
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense

import metric
from src import config
from src.convolution.database_sequence import DatabaseSequence


def load_vgg16_model():
    model = VGG16(weights='imagenet', include_top=False)
    return model


def build_nn(classes_cnt):
    base_model = load_vgg16_model()

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes_cnt, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def tune_nn(model, base_model, sequence):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    model.fit_generator(sequence, epochs=config.max_epochs)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    # last block => block no 5
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    stop_callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required,
                                            patience=config.max_no_improvement_epochs)

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=[metric.top_1_accuracy,
                           metric.top_5_accuracy])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(sequence,
                        epochs=config.max_epochs,
                        callbacks=[stop_callback])

    return model


def evaluate_model(model, sequence):
    scores = model.evaluate_generator(sequence)
    for i in range(len(scores)):
        print("{name}: {value}".format(name=model.metrics_names[i],
                                       value=scores[i]))


def main():
    logging.basicConfig(filename="base_truth.log", level=logging.DEBUG)

    seq_train = DatabaseSequence(config.resources_path + 'preprocessed.hdf5', 16)

    model, base_model = build_nn(50)
    model = tune_nn(model, base_model, seq_train)

    seq_train.close()
    seq_test = DatabaseSequence(config.resources_path + 'preprocessed.hdf5', 16)

    evaluate_model(model, seq_test)

    seq_test.close()

    return 0


if __name__ == "__main__":
    main()
