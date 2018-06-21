import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, CSVLogger

# Save Bottleneck Features ?
save_features = 1

# Suffix (eg slice_1 etc)
suffix = 'distributed_slice_rezised_augment'

# dimensions of our images.
img_width, img_height = 150, 150

log_path = 'results/bottleneck_fc_model_' + suffix + '.log'
top_model_weights_path = 'weights/bottleneck_fc_model_' + suffix + '.h5'
train_data_dir = 'data/' + suffix + '/train'
validation_data_dir = 'data/' + suffix + '/validation'

# needs to be separable by 3 (no Rest)
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

epochs = 25
batch_size = 32


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save("features/bottleneck_features_train_" + suffix, bottleneck_features_train)
    np.load("features/bottleneck_features_train_" + suffix + ".npy")

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
np.save("features/bottleneck_features_validation_" + suffix, bottleneck_features_validation)
np.load("features/bottleneck_features_validation_" + suffix + ".npy")


def train_top_model():
# load training data
train_data = np.load('features/bottleneck_features_train_' + suffix + '.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 3) + [1] * (nb_train_samples // 3) + [2] * (nb_train_samples // 3))
    train_labels = to_categorical(train_labels)

# load validation data
validation_data = np.load('features/bottleneck_features_validation_' + suffix + '.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 3) + [1] * (nb_validation_samples // 3) + [2] * (nb_validation_samples // 3))
    validation_labels = to_categorical(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=
    			train_data.shape[1:]))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    csv_logger = CSVLogger(log_path)

    callbacks_list = [checkpoint, csv_logger]

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks_list,
              validation_data=(validation_data, validation_labels))
    # model.save_weights(top_model_weights_path)


if nb_train_samples % 3 == 0 & nb_validation_samples % 3 == 0:
    print("save_bottlebeck_features")
    if save_features:
        save_bottlebeck_features()
    print("train_top_model")
    train_top_model()
else:
    print("Train Samples and Validation Samples need to be separable by 3 (no Rest)")
    print("Train Samples: " + str(nb_train_samples) + " % 3 = " + str(nb_train_samples % 3))
    print("Validation Samples: " + str(nb_validation_samples) + " % 3 = " + str(nb_validation_samples % 3))
