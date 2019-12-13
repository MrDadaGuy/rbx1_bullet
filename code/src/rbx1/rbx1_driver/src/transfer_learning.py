#! /usr/bin/env python

import glob, time, os, multiprocessing
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from datagen import DataGenerator
import matplotlib.pyplot as plt
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"       # supposedly fixes multiprocessing locking error on hdf5 checkpoint writing

params = {'dim' : (256, 256),
#            'batch_size' : 24,      # 24 = val_loss: 0.0033
            'n_channels' : 3,
            'shuffle' : False
}

batch_sizes = (8, 16, 32, 64, 128, 256)
run_epochs = 200
num_gpus = 1
lr_patience = 3
stop_patience = lr_patience + 2
run_timestamp = str(time.time())
num_workers = multiprocessing.cpu_count() - 2

image_file_names = glob.glob("files/*.png")       # gets the image names including relative path
text_file_names = []
for image in image_file_names:
    text_file_names.append(image[:-4] + ".txt" )      # get the corresponding text file

# partition the FILE NAMES into training and testing splits using 80% of the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(image_file_names, text_file_names, test_size=0.2, random_state=42)

for bs in batch_sizes:

   print("[INFO] starting loop for batch size = {}...".format(bs))

   early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=stop_patience)
   mcp_save = ModelCheckpoint(str(time.time()) + '.chkpt.hd5', save_best_only=True, monitor='val_loss', mode='min')
   reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, verbose=1, epsilon=1e-4, mode='min')
   tensor_board1 = TensorBoard(log_dir='./logs/{}_{}_1'.format(run_timestamp, bs), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
   tensor_board2 = TensorBoard(log_dir='./logs/{}_{}_2'.format(run_timestamp, bs), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

   training_generator = DataGenerator(trainX, trainY, batch_size=bs, **params)
   validation_generator = DataGenerator(testX, testY, batch_size=bs, **params)

   # create the base pre-trained model
   input_tensor = Input(shape=(params['dim'][0], params['dim'][1], params['n_channels']))  #  make sure inception model takes my image sizes
   base_model = InceptionV3(weights='imagenet', input_tensor=input_tensor, include_top=False)

   # add a global spatial average pooling layer
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(1024, activation='relu')(x)
   predictions = Dense(4, activation='linear')(x)

   # this is the model we will train
   model = Model(inputs=base_model.input, outputs=predictions)

   if num_gpus > 1:
      model = multi_gpu_model(model, gpus=num_gpus, cpu_merge=True, cpu_relocation=True)  #cpu_relocation False by default

   # first: train only the top layers (which were randomly initialized)
   # i.e. freeze all convolutional InceptionV3 layers
   for layer in base_model.layers:
      layer.trainable = False

   # compile the model (should be done *after* setting layers to non-trainable)
   model.compile(optimizer=Adam(lr=0.001), loss='mse') #, momentum=0.9), loss='mean_absolute_error')       # was "mse"
   #model.compile(optimizer='adam', loss='mean_absolute_error')         # was "mse"

   # train the model on the new data for a few epochs
   print("[INFO] training model...")
   #model.fit(trainX, trainY, validation_data=(testX, testY), epochs=run_epochs, batch_size=8) #, callbacks=[early_stopping])

   H1 = model.fit_generator(generator=training_generator,
                     validation_data=validation_generator,
                     use_multiprocessing=True,
                     workers=num_workers, 
                     epochs=run_epochs, 
                     callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensor_board1])

   model.save("{}_{}_tl_1.hd5".format(run_timestamp, bs))

   # at this point, the top layers are well trained and we can start fine-tuning
   # convolutional layers from inception V3. We will freeze the bottom N layers
   # and train the remaining top layers.

   # let's visualize layer names and layer indices to see how many layers
   # we should freeze:
   for i, layer in enumerate(base_model.layers):
      print(i, layer.name)

   # we chose to train the top 2 inception blocks, i.e. we will freeze
   # the first 249 layers and unfreeze the rest:
   for layer in model.layers[:249]:
      layer.trainable = False
   for layer in model.layers[249:]:
      layer.trainable = True


   # we need to recompile the model for these modifications to take effect
   model.compile(optimizer=Adam(lr=0.0001), loss='mse') #, momentum=0.9), loss='mean_absolute_error')       # was "mse"

   # we train our model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers
   H2 = model.fit_generator(generator=training_generator,
                     validation_data=validation_generator,
                     use_multiprocessing=True,
                     workers=num_workers, 
                     epochs=run_epochs, 
                     callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensor_board2])

   model.save("{}_{}_tl_2.hd5".format(run_timestamp, bs))

   H1 = H1.history
   H2 = H2.history

   for H in [H1, H2]:
      N = np.arange(0, len(H["loss"]))
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(N, H["loss"], label="train_loss")
      plt.plot(N, H["val_loss"], label="test_loss")
   #    plt.plot(N, H["acc"], label="train_acc")
   #    plt.plot(N, H["val_acc"], label="test_acc")
      plt.title("Tranfser Learning")
      plt.xlabel("Epoch #")
      plt.ylabel("Loss/Accuracy")
      plt.legend()
      
      # save the figure
      plt.savefig("plot_{}_{}.png".format(run_timestamp, bs))
      plt.close()
