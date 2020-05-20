import numpy as np
import tensorflow as tf
import cv2

class NeuralNet(object):
	def __init__(self):
	  mnist = tf.keras.datasets.mnist
	  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
	  training_images=training_images.reshape(60000, 28, 28, 1)
	  training_images=training_images / 255.0
	  test_images = test_images.reshape(10000, 28, 28, 1)
	  test_images=test_images/255.0
	  self.model = tf.keras.models.Sequential([
	    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
	    tf.keras.layers.MaxPooling2D(2, 2),
	    tf.keras.layers.Flatten(),
	    tf.keras.layers.Dense(128, activation='relu'),
	    tf.keras.layers.Dense(10, activation='softmax')
	  ])
	  self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	  self.model.fit(training_images, training_labels, epochs=10)

	def predict(self, image):
		input = cv2.resize(image, (28 , 28)).reshape((28 , 28,1)).astype('float32') / 255
		return self.model.predict_classes(np.array([input]))