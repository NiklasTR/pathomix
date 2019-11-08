from keras.layers import Dense
from keras.models import Model
import os

def train_ultimate_layers(model,
                          train_generator,
                          validation_generator,
                          steps_per_epoch_train,
                          steps_per_epoch_val,
                          out_path,
                          epochs=20,
                          num_of_dense_layers=0,
                          dense_layer_dim=32,
                          tensor_board_callback=None,
                          bsave=False):
        # remove final layer
        # this will remove final classification layer
        # new last layer has shape (after GlobalAveragepooling2d) (None, 1280)
        model.layers.pop()

        # freeze all layers in pretrained model
        for l in model.layers:
            l.trainable = False

        # add 2 fully connected layers
        # on toy data the additional Dense(32) layer improved the accuracy for the validation set after the same number of epochs from
        # 0.9571 to 0.9831. No over fitting was observed
        x = model.output
        for dl in range(num_of_dense_layers):
                x = Dense(dense_layer_dim)(x)
        pred = Dense(1, activation='sigmoid')(x)

        mymodel = Model(inputs=model.input, outputs=pred)

        # sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.0)
        mymodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', auroc])
        #mymodel.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        mymodel.fit_generator(
                train_generator,
                steps_per_epoch=steps_per_epoch_train,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=steps_per_epoch_val,
                callbacks=[tensor_board_callback])

        if bsave:
                # serialize model to JSON
                model_json = mymodel.to_json()
                with open(os.path.join("{}.json".format(out_path)), "w") as json_file:
                        json_file.write(model_json)
                # save weights
                mymodel.save_weights(os.path.join("{}.h5".format(out_path)))
                print('model saved at {}'.format(out_path))

        return mymodel