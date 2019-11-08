import tensorflow as tf

from keras.metrics import binary_accuracy

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, ModelCheckpoint


class roc_callback(Callback):
    def __init__(self):
        self.x = []
        self.y = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        print('auc {}'.format(roc))
        return

    def on_batch_begin(self, batch, logs={}):
        self.x.append(batch[0])
        self.y.append(batch[1])
        return

    def on_batch_end(self, batch, logs={}):
        return


def model_checkpointer():
    checkpointer = ModelCheckpoint(filepath='/weights.hdf5', verbose=1, save_best_only=True, period=5)
    return checkpointer

'''
class CumulativeHistory(History):
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            super(CumulativeHistory, self).on_train_begin(logs)
'''

class LossHistory(keras.callbacks.Callback):
    def on_epoche_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def get_accuracy(y_true, y_pred):
    return binary_accuracy(y_true, y_pred)


def get_auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)