#! /usr/bin/env python

import random, time, pickle, os, argparse, sys
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
from generate_samples import generate
import matplotlib.pyplot as plt
import numpy as np
import cv2

matplotlib.use("Agg")

np.set_printoptions(threshold=np.sys.maxsize)

# initialize the data and labels
data = []
labels = []

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", default="model.mdl", help="path to output model")
#ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (128, 128, 3)
 
num_samples = 1024			# how many image samples to generate

data, labels = generate(num_samples, IMAGE_DIMS[1], IMAGE_DIMS[0])

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
 
# partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation	# dont think we want rotated images, thats going to introduce problems
aug = ImageDataGenerator(rotation_range=0, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.05, zoom_range=0.0,
	horizontal_flip=False, fill_mode="nearest")		

num_classes = 6
# initialize the model
print("[INFO] compiling model...w={}, h={}, d={}".format(IMAGE_DIMS[1], IMAGE_DIMS[0], IMAGE_DIMS[2]))
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=num_classes)
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])

#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=2)  # KerasRegressor for regression problem



# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), 
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=0, shuffle=False)	# was verbose=1

#H = model.fit(x=trainX, y=trainY, batch_size=BS, validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
 
# save the label binarizer to disk
#print("[INFO] serializing label binarizer...")
#f = open(args["labelbin"], "wb")
#f.write(pickle.dumps(lb))
#f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])