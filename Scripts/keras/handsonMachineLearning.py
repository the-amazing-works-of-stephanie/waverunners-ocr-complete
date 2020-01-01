import keras
from keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
X_train_full[500:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# class_names[y_train[0]]

# creates a Sequential model, which is a good kind for neural networks that contain a single stack of layers
# MaxPool: pool size of 2; divides each spatial dimension by a factor of 2
# Flatten: build the first layer and add it to model; converts each image into a 1D array
# Dense: contains 128 and 64 neurons. Each Dense layer manages its own weight matrix, containing all connection weights
# between neurons and their inputs. Also manages a vector of bias terms
# Dense: contains 10 neurons (one per class) using the softmax activation function

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")])
"""
keras.layers.Conv2D(64, 7, activation="relu", padding="same",input_shape=[28,28,1]),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
"""


# displays all the model's layers
# model.summary()

# print(model.layers)
# hidden1 = model.layers[1]
# print(hidden1.name)
#
# weights, biases = hidden1.get_weights()
# # weights should be initialized at random (in order to break symmetry)
# print(weights)
# print(weights.shape)
# # biases are initialized at zero
# print(biases)
# print(biases.shape)

# specify the loss function & which optimizer to use, plus extra metrics
# sparse_categorical_crossentropy is used because we only have a target class index (0 - 9). see pg 375 for more details
# "sgd" optimizer means simple Stochastic Gradient Descent
# since we're training a classifier, it's best to measure metrics by "accuracy"
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# .fit() trains the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

model.save("stephanieTest2_112719.h5")

# model.evaluate(X_test, y_test)
#
# X_new = X_test[:3]
# y_proba = model.predict(X_new)
# y_proba.round(2)
