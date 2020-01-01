import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt


class FindContours:
    """
    This function processes the image
    @param filename:
    """
    def processImage(self, filename):
        contour_list = []
        img = cv2.imread(filename)

        # smoothing the image
        # imageblur = cv2.GaussianBlur(img, (5,5),0)
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

        # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        image, contours, hierarchy = cv2.findContours(mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # Create bounding rectangles
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 1)
            C1 = ContourHolder(x - 2, y - 2, w + 4, h + 4)
            contour_list.append(C1)

        self.sortContour(contour_list)
        return contour_list, img

    """
    This function sorts the contours that opencv detects using their x coordinates
    @param Contour_List: 
    """
    def sortContour(self, Contour_List):
        for ch in range(len(Contour_List)):

            min_index = ch
            for j in range(ch + 1, len(Contour_List)):
                if Contour_List[min_index].getX() > Contour_List[j].getX():
                    min_index = j

            Contour_List[ch], Contour_List[min_index] = Contour_List[min_index], Contour_List[ch]

    """This function prepares the image to be processed by correcting size, shape, and color of each temporary PNG image file
    @param image: 
    @return 
    """
    def prepareDigit(self, image):
        img = cv2.imread(image)
        # digits are usually 28 pixels by 28 pixels
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, [1, 28, 28, 1])

        return img

    """
    This function shows image with contours
    @param image: 
    """
    ### NOT BEING CALLED ###
    def showPicture(self, image):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    """
    This function stores the pictures to be used for later
    @param contour_list: 
    @param image: 
    @return 
    """
    def createPictureList(self, contour_list, image):
        count = 0
        for c in contour_list:
            count += 1
            roi = image[c.getY():c.getY() + c.getHeight(), c.getX():c.getX() + c.getWidth()]
            cv2.imwrite(str(count) + '.png', roi)
        return count

    """
    This function clears all temporary image files
    @param digits: 
    """
    def cleanFolder(self, digits):
        for number in range(1, digits + 1):
            os.remove('{0}.png'.format(number))

    """
    This function predicts the numbers as listed in the cropped image file
    @param digits: 
    @param model: 
    @return 
    """
    def processContours(self, digits, model):
        results = []
        for number in range(1, digits + 1):
            image = self.prepareDigit('{0}.png'.format(number))
            #showPicture(image)
            # execute model to read digits
            output = model.predict(image)
            # return indices of the max element of the array in a particular axis
            results.append(np.argmax(output))

        # Returns the predicted numbers
        return results



# This class stores information regarding bounding boxes around contours
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