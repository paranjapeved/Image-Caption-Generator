
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import pickle

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
#This returns a list of all image identifiers in the file containing the list of images to be trained on
def load_img_names(filename):
	doc = load_doc(filename)
	img_names = []
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		img_names.append(identifier)
	return set(img_names)

#This returns a dictionary containg the image_id -> list of descriptions to each image_id
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = {}
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = []
			# put "startseq" at the beginning of the descriptions and "endseq" at the end
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# Append a new description string to the list in the dictionary
			descriptions[image_id].append(desc)
	return descriptions

# Returns an object of features of an image from features.pkl
#dataset is the file which contains the names of images to be trained on
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# Puts the descriptions into a list of words
def to_words(descriptions):
	all_desc = []
	for key in descriptions.keys():
		for d in descriptions[key]:
			all_desc.append(d)
	return all_desc


# Maps the words to integers
def create_tokenizer(descriptions):
	words = to_words(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(words)
	return tokenizer


#Returns the max length of a description
def max_length(descriptions):
	lines = to_words(descriptions)
	return max(len(d.split()) for d in lines)

#Creates batches of Image, Text_input and Text_output
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)


def define_model(vocab_size,max_len):
	#Dense model For image embeddings
	input1 = Input(shape=(4096,))
	dropout1 = Dropout(0.5)(input1)
	fc1 = Dense(256,activation='relu')(dropout1)

	#Sequence Model For text
	input2 = Input(shape=(max_len,))
	emb = Embedding(vocab_size,256)(input2)
	dropout2 = Dropout(0.5)(emb)
	fc2 = LSTM(256)(dropout2)

	#Decoder
	decoder1 = add([fc2, fc1])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[input1, input2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_doc(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

tokenizer = create_tokenizer(train_descriptions)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
