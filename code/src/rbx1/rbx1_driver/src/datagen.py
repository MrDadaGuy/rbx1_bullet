import numpy as np
import keras
from keras.preprocessing.image import img_to_array
import cv2
#from generate_samples import read_from_directory

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(128,128), n_channels=3, n_classes=None, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp, use3d=False):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
		y = np.empty((self.batch_size), dtype=object) #dtype=np.float32

		cv_image = None
		label_list = []

#        print("[INFO] - Using IDs {}".format(list_IDs_temp))

		for i, ID in enumerate(list_IDs_temp):
			cv_image = cv2.imread(ID)
			image = cv2.resize(cv_image, (self.dim))	# use OpenCV to resize
			image = img_to_array(image)			# convert to numpy using keras preprocessing
			X[i,] = image
			textfile = ID[:-4] + ".txt"
			with open (textfile, 'r') as f:
				temp = f.read()
				content = (map(float, temp[1:-1].split(",")))
				if not use3d:
					content = content[:2] + content[3:5]        # NOTE:  taking out the Z axis
				label_list.append(content)

		X = np.array(X, dtype="float") / 255.0
		y = np.stack(label_list, axis=0)

		return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)