from os import listdir
from pickle import dump
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from numpy import argmax

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

def get_features(filename):
	# model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
	image = load_img(filename,target_size=(224,224))
	image = img_to_array(image)
	print(image.shape)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
	image = preprocess_input(image)
		# get features
	img_feature = model.predict(image, verbose=0)
	return img_feature

if __name__ == '__main__':
    path = 'dog.jpg'
    features = get_features(path)
    with open('tokenizer.pickle', 'rb') as handle:
        token = pickle.load(handle)
    model = load_model('captiongenerator.h5')
    text = generate_desc(model,token,features,34)
    print(text)
