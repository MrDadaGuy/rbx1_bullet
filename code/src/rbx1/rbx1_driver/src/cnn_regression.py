#! /usr/bin/env python

import os, sys, time, multiprocessing, re, glob
import simplejson as json
import numpy as np
#from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
import models
from generate_samples import generate
from datagen import DataGenerator

#np.set_printoptions(threshold=np.sys.maxsize)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"       # supposedly fixes multiprocessing locking error on hdf5 checkpoint writing

params = {'dim' : (256, 256),
			'n_channels' : 3,
			'shuffle' : False
}
run_epochs = 50
num_gpus = 1
num_threads = multiprocessing.cpu_count() - 2
lr_patience = 3
stop_patience = lr_patience + 2

# get list of files that we're going to use for our data and labels
image_file_names = glob.glob("files/*.png")       # gets the image names including relative path
text_file_names = []
for image in image_file_names:
	text_file_names.append(image[:-4] + ".txt" )      # get the corresponding text file

# partition the FILE NAMES into training and testing splits using 80% of the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(image_file_names, text_file_names, test_size=0.2, random_state=42)

batch_sizes = [8, 16, 32, 64]		# 8, 16, 32, 64   4, 128
optimizers = ["adam"]			# , "SGD", "adadelta"
loss_functions = ["mse"]		# , "msle", "mae", "mape", "kld", "cosine", 
filter_options = [(16,32,64), (8,16), (16,32), (32,64), (16,64), (64,128), (8,16,32,64), (16,32,64,128), (64, 128, 256, 512), (8,16,32,64,128)]				# (8,16), (16,32), (32,64), (16,64), (64,128), (16,32,64), (8,16,32,64), (16,32,64,128), (8,16,32,64,128)
#filter_options = [(16,32,64)]				# (8,16), (16,32), (32,64), (16,64), (64,128), (16,32,64), (8,16,32,64), (16,32,64,128), (8,16,32,64,128)

time_stamp = str(time.time())
try:
	os.makedirs("./models/{}".format(time_stamp))	# create directory for checkpoint saving.. tensorboard is smart enough to create its own log subdirs
except OSError:
	pass

for bs in batch_sizes:
	for opt in optimizers:
		for lsfn in loss_functions:
			for filtopt in filter_options:
				arch = "vgg" # "scnn" "cscnn"
				this_run = "{}_{}_{}_{}_{}".format(arch, bs, opt, lsfn, str(filtopt).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_"))

				early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=stop_patience, min_delta=0.0001, restore_best_weights=True)
				mcp_save = ModelCheckpoint("./models/{}/{}.chkpt.hd5".format(time_stamp, this_run), save_best_only=True, monitor='val_loss', mode='min')		# verbose=1, 
				reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, verbose=1, epsilon=1e-4, mode='min')
				tensor_board = TensorBoard(log_dir="./logs/{}/{}".format(time_stamp, this_run), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

				training_generator = DataGenerator(trainX, trainY, batch_size=bs, **params)
				validation_generator = DataGenerator(testX, testY, batch_size=bs, **params)

				if arch == "scnn":
					model = models.simple_cnn(params['dim'][0], params['dim'][1], params['n_channels'], filters=filtopt, outputs=4, regress=True)		#model = models.create_cnn(64, 64, 3, regress=True)
				elif arch == "cscnn":
					model = models.comically_simple_cnn(params['dim'][0], params['dim'][1], params['n_channels'], outputs=4, filter=32)
				elif arch == "vgg":
					model = models.smaller_vgg(params['dim'][0], params['dim'][1], params['n_channels'], 4)     # NOTE:  was 6, getting rid of Z axis..

				if num_gpus > 1:
					model = multi_gpu_model(model, gpus=num_gpus, cpu_merge=True, cpu_relocation=True)  #cpu_relocation False by default

				model.compile(loss=lsfn, optimizer=opt)
				print("[INFO] This Run = {}".format(this_run))
				print("[INFO] Training model...")
				history = model.fit_generator(generator=training_generator,
									validation_data=validation_generator,
									use_multiprocessing=True,
									workers=num_threads, 
									epochs=run_epochs, 
									callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensor_board])

				best_val_loss = min(history.history["val_loss"])

				summary_file = "./models/{}/summary.txt".format(time_stamp)
				with open(summary_file, "a") as f:
					f.write("{} best val_loss: {}\n".format(this_run, best_val_loss))

				print("[INFO] Saving model...")
				model.save("./models/{}/{}.hd5".format(time_stamp, this_run))
