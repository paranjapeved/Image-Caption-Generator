# Image-Caption-Generator
Image Caption Generator using CNN and RNN

Dataset used:
This model is trained on the Flickr8k dataset of images and captions

Instructions to run:
The create_embeddings.py file creates embeddings for the images using a pre-trained inception net included in keras.
Then run the main_deep.py to train the multi-layered RNN
To predict your images using the REST API, run the predict.py which would start the service. Then pass the image as 
curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'


