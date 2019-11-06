import tensorflow.keras as keras
from keras_mixnets import MixNetSmall 
import tensorflow as tf

class RestoreCkptCallback(keras.callbacks.Callback):
    def __init__(self, pretrained_model_path):
        self.pretrained_file = pretrained_model_path
        self.sess = keras.backend.get_session()
        self.saver = tf.train.Saver()
    def on_train_begin(self, logs=None):
        if self.pretrained_model_path:
            self.saver.restore(self.sess, self.pretrained_model_path)
            print('load weights: OK.')


model = MixNetSmall((224, 224, 3), include_top=True) 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

restore_ckpt_callback = RestoreCkptCallback(pretrained_model_path='/home/ubuntu/pathomix/mix_models/mixnet-s/model.ckpt.data-00000-of-00001')

#print(restore_ckpt_callback)
