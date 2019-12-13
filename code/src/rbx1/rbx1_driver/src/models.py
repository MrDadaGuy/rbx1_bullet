#! /usr/bin/env python

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))
 
	# check to see if the regression node should be added
	if regress:
		model.add(Dense(6, activation="linear"))
 
	# return our model
	return model


def smaller_vgg(width, height, depth, classes=None):
	# initialize the model along with the input shape to be
	# "channels last" and the channels dimension itself
	model = Sequential()

#	print("Classes = {}".format(classes))

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1
		print("Channels First")
	else:
		inputShape = (height, width, depth)
		chanDim = -1
		print("Channels Last")

	# CONV => RELU => POOL
	model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Dropout(0.25))

	# (CONV => RELU) * 2 => POOL
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# (CONV => RELU) * 2 => POOL
	model.add(Conv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# output
	model.add(Dense(classes))
	model.add(Activation("linear"))

	model.summary()

	return model


def simple_cnn(width, height, depth, outputs=4, filters=(16, 32, 64), regress=False):		# , fc_sizes=None
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

    	# define the model input
	inputs = Input(shape=inputShape)
 
	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs
 
		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

   	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(filters[-1])(x)		#	Use the last CNN size for first dense layer size  -- x = Dense(64)(x)        #	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
 
	# apply another FC layer, this one to match the number of nodes coming out of the MLP
	x = Dense(filters[0])(x)				#x = Dense(16)(x)            # 	x = Dense(4)(x)
	x = Activation("relu")(x)
 
	# check to see if the regression node should be added
	if regress:
		x = Dense(outputs, activation="linear")(x)
 
	# construct the CNN
	model = Model(inputs, x)
#	model.summary()

	# return the CNN
	return model

def comically_simple_cnn(width, height, depth, outputs=4, filter=32):
	conv_size = filter
	inputShape = (height, width, depth)
	chanDim = -1
	inputs = Input(shape=inputShape)
	x = Conv2D((conv_size), (3, 3), padding="same")(inputs)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)
	x = Flatten()(x)
#	x = Dense(64, activation="relu")(x)
	x = Dense(outputs, activation="linear")(x)
	model = Model(inputs, x)
	model.summary()
	return model
