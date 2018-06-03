import logging

from keras import Model, callbacks
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D

from src import config
from src import file
from src import metric
from src.convolution.database_sequence import DatabaseSequence


def load_vgg16_model():
    model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
    return model


def build_nn(classes_cnt):
    base_model = load_vgg16_model()

    x = base_model.output

    # classifier
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(classes_cnt, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def tune_nn(model, base_model, train_seq, validation_seq):
    """
    ref url: https://keras.io/applications/
    """
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    stop_callback = callbacks.EarlyStopping(min_delta=config.min_improvement_required,
                                            patience=config.max_no_improvement_epochs)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[metric.top_1_accuracy,
                           metric.top_5_accuracy])
    model.fit_generator(train_seq,
                        validation_data=validation_seq,
                        epochs=config.max_epochs,
                        callbacks=[stop_callback])

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from VGG16. We will freeze the bottom N layers
    # and train the remaining top layers.

    # we chose to train the top 1 VGG16 blocks, i.e. we will freeze
    # the first 15 layers and unfreeze the rest:
    # last block => block no 5
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True

    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=[metric.top_1_accuracy,
                           metric.top_5_accuracy])

    # we train our model again (this time fine-tuning the top 1 VGG16 blocks
    # alongside the top Dense layers
    model.fit_generator(train_seq,
                        validation_data=validation_seq,
                        epochs=config.max_epochs,
                        callbacks=[stop_callback])

    return model


def evaluate_model(model, test_seq):
    scores = model.evaluate_generator(test_seq)
    for i in range(len(scores)):
        print("{name}: {value}".format(name=model.metrics_names[i],
                                       value=scores[i]))


def main():
    logging.basicConfig(filename="base_truth.log", level=logging.DEBUG)

    total_cls_cnt = file.get_total_cls_cnt(config.set_path)
    assert total_cls_cnt != 0
    seq_train = DatabaseSequence(config.get_convolution_datasets_path('training'), config.batch_size, total_cls_cnt,
                                 True)
    seq_validation = DatabaseSequence(config.get_convolution_datasets_path('validation'),
                                      config.batch_size, total_cls_cnt, True)

    model, base_model = build_nn(total_cls_cnt)
    model = tune_nn(model, base_model, seq_train, seq_validation)

    seq_validation.close()
    seq_train.close()
    seq_test = DatabaseSequence(config.get_convolution_datasets_path('test'), config.batch_size, total_cls_cnt, True)

    evaluate_model(model, seq_test)

    seq_test.close()

    return 0


if __name__ == "__main__":
    main()
