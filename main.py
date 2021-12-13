import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# paths
PATH = './data/'
RAW_BASE_PATH = PATH + 'raw_data/'
SPECTOGRAM_BASE_PATH = PATH + 'spectogram_data/'
NUM_CLASSES = 8
AUDIO_LEN = 32000

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# load raw data
base_path = RAW_BASE_PATH
raw_train_X = load(base_path + 'train_x.pck')
raw_train_Y = load(base_path + 'train_y.pck')
raw_test_X = load(base_path + 'test_x.pck')
raw_test_Y = load(base_path + 'test_y.pck')
raw_valid_X = load(base_path + 'valid_x.pck')
raw_valid_Y = load(base_path + 'valid_y.pck')

raw_train_X = raw_train_X[:, :AUDIO_LEN]
raw_test_X = raw_test_X[:, :AUDIO_LEN]
raw_valid_X = raw_valid_X[:, :AUDIO_LEN]

raw_train_X = raw_train_X.reshape([-1, AUDIO_LEN, 1])
raw_test_X = raw_test_X.reshape([-1, AUDIO_LEN, 1])
raw_valid_X = raw_valid_X.reshape([-1, AUDIO_LEN, 1])

assert raw_train_X.shape[0] == raw_train_Y.shape[0]
assert raw_test_X.shape[0] == raw_test_Y.shape[0]
assert raw_valid_X.shape[0] == raw_valid_Y.shape[0]

# load spectogram data
base_path = SPECTOGRAM_BASE_PATH
spec_train_X = load(base_path + 'spec_train_x.pck')
spec_train_Y = load(base_path + 'spec_train_y.pck')
spec_test_X = load(base_path + 'spec_test_x.pck')
spec_test_Y = load(base_path + 'spec_test_y.pck')
spec_valid_X = load(base_path + 'spec_valid_x.pck')
spec_valid_Y = load(base_path + 'spec_valid_y.pck')

spec_train_X = spec_train_X[:, :, :600]
spec_test_X = spec_test_X[:, :, :600]
spec_valid_X = spec_valid_X[:, :, :600]

spec_train_X = spec_train_X.reshape([-1, 64, 600, 1])
spec_test_X = spec_test_X.reshape([-1, 64, 600, 1])
spec_valid_X = spec_valid_X.reshape([-1, 64, 600, 1])

assert spec_train_X.shape[0] == spec_train_Y.shape[0]
assert spec_test_X.shape[0] == spec_test_Y.shape[0]
assert spec_valid_X.shape[0] == spec_valid_Y.shape[0]


values = np.array(['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

def plot_accuracy(history, model_name):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name + "_acc.png")


def plot_loss(history, model_name):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name + "_loss.png")

# https://arxiv.org/pdf/1610.00087.pdf
def make_convnet_model1():
    fsize = 3
    weight_decay = 0.0001

    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv1D(64, kernel_size=80, strides=4, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay), input_shape=(AUDIO_LEN, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 2
        keras.layers.Conv1D(64, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 3
        keras.layers.Conv1D(64, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 4
        keras.layers.Conv1D(128, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 5
        keras.layers.Conv1D(128, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 6
        keras.layers.Conv1D(256, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 7
        keras.layers.Conv1D(256, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 8
        keras.layers.Conv1D(512, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 9
        keras.layers.Conv1D(512, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # Dense layer
        keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
        # keras.layers.Flatten(),
        # keras.layers.Dropout(dropout),
        # keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        # keras.layers.Dropout(dropout),
        # keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        # keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_convnet_model2():
    fsize = 3
    weight_decay = 0.0001


    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv1D(128, kernel_size=80, strides=4, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=weight_decay), input_shape=(AUDIO_LEN, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 2
        keras.layers.Conv1D(128, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 3
        keras.layers.Conv1D(256, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 4
        keras.layers.Conv1D(512, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # Dense layer
        keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_convnet_model3():
    fsize = 3
    weight_decay = 0.0001
    dropout = 0.5

    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv1D(64, kernel_size=80, strides=4, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay), input_shape=(AUDIO_LEN, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 2
        keras.layers.Conv1D(64, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 3
        keras.layers.Conv1D(64, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 4
        keras.layers.Conv1D(128, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 5
        keras.layers.Conv1D(128, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 6
        keras.layers.Conv1D(256, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 7
        keras.layers.Conv1D(256, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 8
        keras.layers.Conv1D(512, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        #keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # layer 9
        keras.layers.Conv1D(512, kernel_size=fsize, strides=1, padding='same', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l=weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling1D(pool_size=4, strides=None),
        # Dense layer
        keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def random_forest():
    train_Y_categorical = get_categories(raw_train_Y)
    valid_Y_categorical = get_categories(raw_valid_Y)

    nsamples, nx, ny = raw_train_X.shape
    new_train_X = raw_train_X.reshape((nsamples, nx * ny))

    nsamples, nx, ny = raw_valid_X.shape
    new_valid_X = raw_valid_X.reshape((nsamples, nx * ny))

    for i in [10, 100, 1000]:
        clf = RandomForestClassifier(n_estimators=i)
        clf.fit(new_train_X, train_Y_categorical)
        y_pred = clf.predict(new_valid_X)
        acc = metrics.accuracy_score(valid_Y_categorical, y_pred)
        print("validation acc", acc)
        path = "./nets/rf_" + str(i) + ".joblib"
        joblib.dump(clf, path)


def make_spectogram_convnet_model1():
    dropout = 0.5
    fsize = 3
    weight_decay = 0.001

    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu', input_shape=(64, 600, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 2
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 3
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 4
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 5
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # flat layer
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_spectogram_convnet_model2():
    dropout = 0.5
    fsize = 5
    weight_decay = 0.001

    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu', input_shape=(64, 600, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 2
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 3
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # flat layer
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_spectogram_convnet_model3():
    dropout = 0.5
    fsize = 3
    weight_decay = 0.001

    model = keras.models.Sequential([
        # layer 1
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu', input_shape=(64, 600, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 2
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 3
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 4
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 5
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # layer 6
        keras.layers.Conv2D(64, fsize, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        # flat layer
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay)),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_labels(data):
    return label_encoder.inverse_transform(np.argmax(data, axis=1))


def get_categories(data):
    x = np.argmax(data, axis=1)
    return x


def random_forest_spec():
    train_Y_categorical = get_categories(spec_train_Y)
    valid_Y_categorical = get_categories(spec_valid_Y)

    nsamples, nx, ny, nz = spec_train_X.shape
    new_train_X = spec_train_X.reshape((nsamples, nx * ny * nz))

    nsamples, nx, ny, nz = spec_valid_X.shape
    new_valid_X = spec_valid_X.reshape((nsamples, nx * ny * nz))

    for i in [10, 100, 1000]:
        clf = RandomForestClassifier(n_estimators=i)
        clf.fit(new_train_X, train_Y_categorical)
        y_pred = clf.predict(new_valid_X)
        acc = metrics.accuracy_score(valid_Y_categorical, y_pred)
        print("validation acc", acc)
        path = "./nets/spec_rf_" + str(i) + ".joblib"
        joblib.dump(clf, path)


def train_model():
    model = make_convnet_model3()
    model.summary()
    checkpoint_filepath = "./nets/convnet3"

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=0.00005,
        verbose=1)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(
        raw_train_X, raw_train_Y,
        validation_data=(raw_test_X, raw_test_Y),
        batch_size=128,
        epochs=50,
        shuffle=True,
        callbacks=[reduce_lr, model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    plot_accuracy(history, checkpoint_filepath)
    plot_loss(history, checkpoint_filepath)
    model.save(checkpoint_filepath + '.h5')
    evaluate_keras_model(model)


def train_spec_model():
    model = make_spectogram_convnet_model1()
    model.summary()
    checkpoint_filepath = "./nets/spec_convnet1"

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=0.00005,
        verbose=1)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    history = model.fit(
        spec_train_X, spec_train_Y,
        validation_data=(spec_valid_X, spec_valid_Y),
        batch_size=64,
        epochs=50,
        callbacks=[reduce_lr, model_checkpoint_callback]
    )
    plot_accuracy(history, checkpoint_filepath)
    plot_loss(history, checkpoint_filepath)
    model.save(checkpoint_filepath + '.h5')
    evaluate_spec_keras_model(model)


def evaluate_keras_model(model):
    train_loss, train_acc = model.evaluate(raw_train_X, raw_train_Y, verbose=0)
    print("Training Accuracy: {:5.2f}%".format(100 * train_acc))

    test_loss, test_acc = model.evaluate(raw_test_X, raw_test_Y, verbose=0)
    print("Testing Accuracy: {:5.2f}%".format(100 * test_acc))

    valid_loss, valid_acc = model.evaluate(raw_valid_X, raw_valid_Y, verbose=0)
    print("Validation Accuracy: {:5.2f}%".format(100 * valid_acc))

    return [train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc]


def evaluate_spec_keras_model(model):
  train_loss, train_acc = model.evaluate(spec_train_X, spec_train_Y, verbose=0)
  print("Training Accuracy: {:5.2f}%".format(100 * train_acc))

  test_loss, test_acc = model.evaluate(spec_test_X, spec_test_Y, verbose=0)
  print("Testing Accuracy: {:5.2f}%".format(100 * test_acc))

  valid_loss, valid_acc = model.evaluate(spec_valid_X,  spec_valid_Y, verbose=0)
  print("Validation Accuracy: {:5.2f}%".format(100 * valid_acc))

  return [train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc]


def main():
    #train_model()
    #random_forest()
    train_spec_model()
    #random_forest_spec()


if __name__ == '__main__':
    main()
