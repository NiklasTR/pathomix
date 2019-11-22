from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
import efficientnet.keras as efn



class EffNetFT:
    def __init__(self,
                 steps_per_epoch_train=None,
                 epochs=None,
                 validation_generator=None,
                 steps_per_epoch_val=None,
                 tensor_board_callback=None,
                 momentum=0.0,
                 nesterov=True,
                 model_path=None):
        self.steps_per_epoch_train = steps_per_epoch_train
        self.epochs = epochs
        self.validation_generator = validation_generator
        self.steps_per_epoch_val = steps_per_epoch_val
        self.tensor_board_callback = tensor_board_callback
        self.momentum = momentum
        self.nesterov = nesterov

        if model_path:
            self.model = self.load_model(model_path)


    def create_model(self):
        return None


    def load_model(self, model_path):

        json_file = open('{}.json'.format(model_path), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        print('load ultimate model from {}'.format(model_path))
        loaded_model.load_weights("{}.h5".format(model_path))
        print("Loaded model from disk")
        self.model = loaded_model

        return loaded_model


    def train(self, assignments):
        lr = assignments['lr']
        decay = assignments['decay']
        for l in self.model.layers:
            l.trainable = True

        sgd = optimizers.SGD(lr=lr, momentum=self.momentum, nesterov=self.nesterov, decay=decay)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch_train,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            validation_steps=self.steps_per_epoch_val,
            callbacks=[self.tensor_board_callback])

        return self.model