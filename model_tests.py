import unittest
import os
from tensorflow import keras
import joblib
import pickle
import numpy as np
from collections import Counter
from sklearn import metrics

PATH_TO_NETS = "./nets/"
PATH_TO_DATA = "./data/spectogram_data/"

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

spec_valid_X = load(os.path.join(PATH_TO_DATA, 'spec_valid_x.pck'))
spec_valid_Y = load(os.path.join(PATH_TO_DATA, 'spec_valid_y.pck'))
spec_valid_X = spec_valid_X[:, :, :600]
spec_valid_X = spec_valid_X.reshape([-1, 64, 600, 1])
assert spec_valid_X.shape[0] == spec_valid_Y.shape[0]

class ModelTester(unittest.TestCase):

    def test_spec_convnet(self):
        path = os.path.join(PATH_TO_NETS, "spec_convnet3.h5")
        model = keras.models.load_model(path)
        valid_loss, valid_acc = model.evaluate(spec_valid_X, spec_valid_Y, verbose=0)
        print('Best ConvNet Spectogram valid accuracy: {:5.2f}%'.format(100 * valid_acc))
        print('Best ConvNet Spectogram valid loss: {:5.2f}'.format(valid_loss))

    def test_ensemble_spec_convnet(self):
        model1 = keras.models.load_model(os.path.join(PATH_TO_NETS, "spec_convnet1.h5"))
        model2 = keras.models.load_model(os.path.join(PATH_TO_NETS, "spec_convnet2.h5"))
        model3 = keras.models.load_model(os.path.join(PATH_TO_NETS, "spec_convnet3.h5"))
        models = [model1, model2, model3]

        results = []
        for model in models:
            prediction = model.predict(spec_valid_X)
            res = np.argmax(prediction, axis=1)
            results.append(res)

        final_predictions = []
        for i in range(len(spec_valid_X)):
            predictions = [prediction[i] for prediction in results]
            data = Counter(predictions)
            key = max(data, key=data.get)
            val = data[key]
            # if all the models have different values go with the value in the third array
            if val == 1:
                key = predictions[2]
            final_predictions.append(key)

        x = (np.array(final_predictions) == np.argmax(spec_valid_Y, axis=1)).sum() / spec_valid_Y.shape[0]
        print('Ensemble ConvNet Spectogram valid accuracy: {:5.2f}%'.format(100 * x))

    def test_rf_spec(self):
        nsamples, nx, ny, nz = spec_valid_X.shape
        new_valid_X = spec_valid_X.reshape((nsamples, nx * ny * nz))
        valid_Y_categorical = np.argmax(spec_valid_Y, axis=1)

        for i in [10, 100, 1000]:
            loaded_model = joblib.load(os.path.join(PATH_TO_NETS, "spec_rf_" + str(i) + ".joblib"))
            y_pred = loaded_model.predict(new_valid_X)
            acc = metrics.accuracy_score(valid_Y_categorical, y_pred)
            print("Random Forest valid accuracy with {} trees: {:5.2f}%".format(i, 100 * acc))


if __name__ == '__main__':
    unittest.main()