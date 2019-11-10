from keras import optimizers
import os
from keras.callbacks import ModelCheckpoint


def fine_tune_model(model,
                   train_generator,
                   validation_generator,
                   steps_per_epoch_train,
                   steps_per_epoch_val,
                   out_path,
                   lr=0.01,
                   decay=1e-6,
                   momentum=0.0,
                   nesterov=False,
                   epochs=20,
                   tensor_board_callback=None,
                   bsave=False):

    callbacks = []
    if tensor_board_callback:
        callbacks.append(tensor_board_callback)

    check_pointer = ModelCheckpoint('best_model_ft_adam.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='max', period=5)

    callbacks.append(check_pointer)

    # unfreeze all layers in pretrained model
    for l in model.layers:
        l.trainable = True

    #sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=nesterov, decay=decay)
    #model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_val,
        callbacks=[tensor_board_callback])

    if bsave:
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join("{}_final.json".format(out_path)), "w") as json_file:
            json_file.write(model_json)
        # save weights
        model.save_weights(os.path.join("{}_final.h5".format(out_path)))
        print('model saved at {}'.format(out_path))

    return model

