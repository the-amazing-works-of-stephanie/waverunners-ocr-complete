import tensorflow as tf
#import input_data
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import math
from scipy import ndimage
from keras.models import load_model


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty

def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def train_and_predict():
    images = np.zeros(4, 784)
    correct_vals = np.zeros((4, 10))
    input_images = 4

    i = 0
    for no in input_images:

        # read the image
        gray = cv2.imread("0" + str(no) + ".png", 0)
        #gray = cv2.imread(no, 0)

        # rescale it
        gray = cv2.resize(255 - gray, (28, 28))
        # better black and white version
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = getBestShift(gray)
        shifted = shift(gray, shiftx, shifty)
        gray = shifted

        # save the processed images
        cv2.imwrite("image_" + str(no) + ".png", gray)
        """
        all images in the training set have an range from 0-1
        and not from 0-255 so we divide our flatten images
        (a one dimensional vector with our 784 pixels)
        to use the same 0-1 based range
        """
        flatten = gray.flatten() / 255.0
        """
        we need to store the flatten image and generate
        the correct_vals array
        correct_val for the first digit (9) would be
        [0,0,0,0,0,0,0,0,0,1]
        """
        images[i] = flatten
        correct_val = np.zeros(10)
        correct_val[no] = 1
        correct_vals[i] = correct_val
        i += 1

    """
    the prediction will be an array with four values,
    which show the predicted number
    """
    #prediction = tf.argmax(y, 1)
    model = load_model('stephanieTest3.h5')

    img = load_image('image_0.png')

    prediction = model.predict(img)
    print(prediction)
    """
    we want to run the prediction and the accuracy function
    using our generated arrays (images and correct_vals)
    """
    # print(sess.run(prediction, feed_dict={x: images, y_: correct_vals}))
    return (sess.run(prediction, feed_dict={x: images, y_: correct_vals}))
    print(sess.run(accuracy, feed_dict={x: images, y_: correct_vals}))


if __name__ == '__main__':
    train_and_predict()
