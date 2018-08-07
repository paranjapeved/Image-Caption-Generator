from os import listdir
from pickle import dump
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

def extract_features(directory,target_size):

	model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	#model = VGG16()
	#Modify model to remove the last layer
	model.layers.pop()
	model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
	features = {}
	print(model.summary())

	for img_name in listdir(directory):
		filename = directory + "/" + img_name

		image = load_img(filename,target_size=(target_size,target_size))
		image = img_to_array(image)

		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		img_feature = model.predict(image, verbose=0)
		# get image id
		image_id = img_name.split('.')[0]
		# store feature
		features[image_id] = img_feature
		print('>%s' % img_name)
	return features


img_features = extract_features("Flicker8k_Dataset",224)

dump(img_features, open('inception.pkl', 'wb'))
