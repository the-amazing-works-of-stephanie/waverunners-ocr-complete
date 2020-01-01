"""
This file is a rip from Daniel's ImageSegmenter code (off of Gitlab). <NOTE: Daniel was one of the original interns from
the summer 2019 intern team.>

I have trained a basic MNIST model and now I need to test how to predict multiple
numbers in one picture.

Notes: stephanieTest1_11_27_29.h5 only seems to rip out 3, 5, and 8.

"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:02:42 2019

@author: Daniel Johsnon
"""

############################ IMAGE SEGMENTATION PORTION ##########################################

# Import libraries
import cv2
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt


# process the image
def processImage(filename):
    contour_list = []
    img = cv2.imread(filename)

    # smoothing the image
    # imageblur = cv2.GaussianBlur(img, (5,5),0)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    # plt.imshow(mean)
    # plt.show()

    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Create bounding rectangles
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 1)
        C1 = ContourHolder(x - 2, y - 2, w + 4, h + 4)
        contour_list.append(C1)

    sortContour(contour_list)
    return contour_list, img


# Sort the contours that opencv detects using their x coordinates
def sortContour(Contour_List):
    for ch in range(len(Contour_List)):

        min_index = ch
        for j in range(ch + 1, len(Contour_List)):
            if Contour_List[min_index].getX() > Contour_List[j].getX():
                min_index = j

        Contour_List[ch], Contour_List[min_index] = Contour_List[min_index], Contour_List[ch]


# Stores information regarding bounding boxes around contours
class ContourHolder(object):

    def __init__(self, xCoord, yCoord, width, height):
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.width = width
        self.height = height

    def getX(self):
        return self.xCoord

    def getY(self):
        return self.yCoord

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    # Stores the pictures to be used for later


def createPictureList(contour_list, image):
    count = 0
    for c in contour_list:
        count += 1
        roi = image[c.getY():c.getY() + c.getHeight(), c.getX():c.getX() + c.getWidth()]
        cv2.imwrite(str(count) + '.png', roi)
    return count


# Show image with contours
def showPicture(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Prepare the image to be processed
def prepareDigit(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, [1, 28, 28, 1])

    return img


def processContours(digits, model):
    results = []
    for number in range(1, digits + 1):
        image = prepareDigit('{0}.png'.format(number))
        output = model.predict(image)
        results.append(np.argmax(output))

    return results


def cleanFolder(digits):
    for number in range(1, digits + 1):
        os.remove('{0}.png'.format(number))


########################################### Process Form ##############################################

from PIL import Image


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


def getDate(image):
    crop(image, (50, 690, 340, 840), 'Date1.jpg')
    crop(image, (50, 875, 340, 1030), 'Date2.jpg')
    crop(image, (50, 1065, 340, 1220), 'Date3.jpg')
    crop(image, (50, 1250, 340, 1420), 'Date4.jpg')
    crop(image, (50, 1440, 340, 1600), 'Date5.jpg')


def getTime(image):
    # Time
    crop(image, (360, 685, 640, 835), 'TIME1.jpg')
    crop(image, (360, 875, 640, 1025), 'TIME2.jpg')
    crop(image, (360, 1065, 640, 1215), 'TIME3.jpg')
    crop(image, (360, 1255, 640, 1405), 'TIME4.jpg')
    crop(image, (360, 1435, 640, 1610), 'TIME5.jpg')


def getPH(image):
    # pH
    crop(image, (660, 675, 945, 847), 'pH1.jpg')
    crop(image, (660, 865, 945, 1036), 'pH2.jpg')
    crop(image, (660, 1055, 945, 1226), 'pH3.jpg')
    crop(image, (660, 1245, 945, 1418), 'pH4.jpg')
    crop(image, (660, 1435, 945, 1600), 'pH5.jpg')


def getTempC(image):
    # tempC
    crop(image, (963, 685, 1250, 850), 'tempC1.jpg')
    crop(image, (963, 875, 1250, 1035), 'tempC2.jpg')
    crop(image, (963, 1065, 1250, 1226), 'tempC3.jpg')
    crop(image, (963, 1245, 1250, 1416), 'tempC4.jpg')
    crop(image, (963, 1445, 1250, 1605), 'tempC5.jpg')


def getAdjustedPH(image):
    # adjusted ph
    crop(image, (1270, 680, 1555, 846), 'adjustedPH1.jpg')
    crop(image, (1270, 870, 1555, 1036), 'adjustedPH2.jpg')
    crop(image, (1270, 1060, 1555, 1225), 'adjustedPH3.jpg')
    crop(image, (1270, 1250, 1555, 1415), 'adjustedPH4.jpg')
    crop(image, (1270, 1440, 1555, 1604), 'adjustedPH5.jpg')


def translatePH(total, model, filename):
    f = open(filename, 'a')
    f.write('PH\n')
    for number in range(1, total + 1):
        filename = 'pH{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        count = createPictureList(contour_list, img)
        resultList = processContours(count, model)
        cleanFolder(count)

        if (len(resultList) == 3):
            resultList[1] = chr(46)
            numbers = '{0}{1}{2}\n'.format(resultList[0], resultList[1], resultList[2])
            f.write(numbers)
        else:
            numbers = '{0}\n'.format(resultList[0])
            f.write(numbers)
    f.close()


def translateTempC(total, model, filename):
    f = open(filename, 'a')
    f.write('TempC\n')
    for number in range(1, total + 1):
        filename = 'tempC{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        count = createPictureList(contour_list, img)
        resultList = processContours(count, model)
        cleanFolder(count)

        f.write('{0}{1}\n'.format(resultList[0], resultList[1]))
    f.close()


def translateAdjustedPH(total, model, filename):
    f = open(filename, 'a')
    f.write('Adjusted PH\n')
    for number in range(1, total + 1):
        filename = 'adjustedPH{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        count = createPictureList(contour_list, img)
        resultList = processContours(count, model)
        cleanFolder(count)

        if (len(resultList) == 3):
            resultList[1] = chr(46)
            numbers = '{0}{1}{2}\n'.format(resultList[0], resultList[1], resultList[2])
            f.write(numbers)
        else:
            numbers = '{0}\n'.format(resultList[0])
            f.write(numbers)
    f.close()


########################################## Master Class #####################################################

class MasterClass:

    def processImage(filename, model, output):
        getTime(filename)
        getPH(filename)
        getTempC(filename)
        getAdjustedPH(filename)

        translatePH(5, model, output)
        translateTempC(5, model, output)
        translateAdjustedPH(5, model, output)


########################################### RUN THE PROGRAM #############################################
import sys


def main(argv):
    model = load_model('stephanieTest3.h5')
    filename = '2961_001.jpg'
    output = 'Result.txt'
    MasterClass.processImage(filename, model, output)


if __name__ == "__main__":
    main(sys.argv)
