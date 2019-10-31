import tensorflow as tf
import tensorflow.keras.preprocessing.image as img
import tensorflow.keras.applications as appl
import tensorflow.keras.callbacks as callb 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, Model

import datetime
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

batch_size = 32
# change in model_nas too
image_size = (224, 224)

class ModelMetrics(callb.Callback):
  
  def on_train_begin(self,logs={}):
    self.precisions=[]
    self.recalls=[]
    self.f1_scores=[]
  def on_epoch_end(self, batch, logs={}):
    
    y_val_pred=self.model.predict_classes(x_val)
   
    _precision,_recall,_f1,_sample=score(y_val,y_val_pred)  
    
    self.precisions.append(_precision)
    self.recalls.append(_recall)
    self.f1_scores.append(_f1)

train_datagen = img.ImageDataGenerator(
        rescale=1./255,
	rotation_range=90,
	vertical_flip=True,
        horizontal_flip=True,
	validation_split=0.
	)

test_datagen = img.ImageDataGenerator(rescale=1./255)

train_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN'
train_samples = []
for r, d, files in os.walk(train_folder):
	for f in files:
		train_samples.append(f)

num_train_samples = len(train_samples)
print(num_train_samples)
train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TEST',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary')

validation_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TEST'
val_samples = []
for r, d, files in os.walk(validation_folder):
        for f in files:
                val_samples.append(f)
num_val_samples = len(val_samples)
print(num_val_samples)
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensor_board_callback = callb.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
metrics = ModelMetrics()


'''
model_nas = appl.NASNetLarge(input_shape=(331, 331, 3), 
			include_top=False, 
			#weights=None,
			weights='imagenet',
			input_tensor=None,
			pooling=None, # irrelevant since include_top=True
			classes=2
			)
'''

model_nas = appl.densenet.DenseNet201(input_shape=(224, 224, 3), 
			include_top=False, 
			#weights=None,
			weights='imagenet',
			input_tensor=None,
			pooling=None, # irrelevant since include_top=True
			classes=2
			)
x = model_nas.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation='softmax')(x)

for layer in model_nas.layers:
	layer.trainable = False

model = Model(inputs=model_nas.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit_generator(
        train_generator,
        #steps_per_epoch=20,
	steps_per_epoch=num_train_samples//batch_size,
        epochs=50,
        validation_data=validation_generator,
        #validation_steps=8, 
	validation_steps=num_val_samples//(batch_size*10), # change for extensive validation
	workers=1,
	use_multiprocessing=False,
	callbacks=[tensor_board_callback])#, metrics])

model.save_weights('first_try.h5')

'''
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
'''
