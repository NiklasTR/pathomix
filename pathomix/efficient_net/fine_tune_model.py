from keras import optimizers
import os

from utils.metrics import get_auroc, get_accuracy, CumulativeHistory
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

    # unfreeze all layers in pretrained model
    for l in model.layers:
        l.trainable = True

    sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=nesterov, decay=decay)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    if tensor_board_callback:
        callbacks.append(tensor_board_callback)

    history = CumulativeHistory()
    callbacks.append(history)

    checkpointer = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)
    callbacks.append(checkpointer)
    accs = []
    aucs = []
    for e in epochs:
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch_train,
            epochs=e+1,
            initial_epoch=e,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch_val,
            callbacks=callbacks)

        y_pred = []
        y_gt = []
        for vs in steps_per_epoch_val:
            x_val ,y_val = validation_generator.ext()
            prediction = model.predict_on_batch(x)
            y_pred.append(prediction)
            y_gt.append(y_val)

        aucs.append(get_auroc(y_gt, y_pred))
        accs.append(get_accuracy(y_gt, y_pred))







    if bsave:
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join("{}.json".format(out_path)), "w") as json_file:
            json_file.write(model_json)
        # save weights
        model.save_weights(os.path.join("{}.h5".format(out_path)))
        print('model saved at {}'.format(out_path))

    return model

