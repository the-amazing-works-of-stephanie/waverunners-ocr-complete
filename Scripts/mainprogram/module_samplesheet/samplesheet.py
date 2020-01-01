from keras.models import load_model
import matplotlib.pyplot as plt


class SampleSheet:
    def __init__(self):
        pass

    """
    This function starts the processing of a user's document.
    @param form: the user's uploaded document
    @param form_type: the type of form the user has indicated
    @param output: the file to write data into from form
    @return the output file
    """
    def startProcessing(self, form, form_type):
        # import keras.models, .hdf data file that is saved in a hierarchical format
        model = load_model('KerasModel4.h5')

        # saves all of function's output into specific file
        output = 'Results.csv'

        # calls the function to start template matching and cropping the entire form
        self.processForm(self, form, model, output, form_type)

        # erases pre-existing contents in output file
        self.erase_file_contents(self, output)

        # second argument is the number of images the program is processing
        self.translatePH(self, 8, model, output)
        # self.translateTempC(self, 8, model, output)
        # self.translateAdjustedPH(self, 8, model, output)
        return output

    """
    This function uses each column on the form, along with a designated template, to locate that column on the form.
    @param form: the user's uploaded document
    """
    def processForm(self, form):
        MatchTemplate.template_match(form, 'BioSolInitpH.jpg', 'bioHeaderCol1.jpg', form_type)
        #crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitpH.jpg")

    """
    This function takes the total number of files, a trained machine learning model, and an output file to read the 
    digits on a file, read them with the model, and write them to a file. 
    @param total: number of files to be processed
    @param model: the training model to read handwritten digits
    @param output: the output file
    """
    def translatePH(self, total, model, output):
        f = open(output, 'a')
        f.write('PH,')
        # for loop is meant for partially naming files
        for number in range(1, total + 1):
            # saves the jpg into a string format
            filename = 'InitpH{0}.jpg'.format(number)
            contour_list, img = FindContours.processImage(filename)
            # the below commands show where the program is making contours
            plt.imshow(img)
            print(contour_list)
            plt.show()
            # createPictureList creates temporary images
            count = FindContours.createPictureList(contour_list, img)
            # calls actual OCR function
            resultList = FindContours.processContours(count, model)
            print("Successful!")
            # deletes temporary PNG files
            FindContours.cleanFolder(count)
            print(resultList)

            # this says if 3 numbers are read
            ### make a test to see WHY just 3 numbers? ###
            ### if len(resultList) <= 6, then loop ###
            # if there are 3 characters, this function inserts a decimal point after the second character
            ### THIS DOESN'T ACCOUNT FOR DIGITS SUCH AS 12.3 ###
            if len(resultList) == 3:
                ### this prints a period based on ASCII table, character #46 is a period ###
                resultList[1] = chr(46)
                # takes the first, second, and third characters and saves to a variable 'numbers', then writes variable to file
                numbers = '{0}{1}{2},'.format(resultList[0], resultList[1], resultList[2])
                f.write(numbers)
            # this accounts for double digit character readings
            if len(resultList) == 2:
                resultList[1] = chr(46)
                numbers = '{0}{1},'.format(resultList[0], resultList[1])
                f.write(numbers)
            # this accounts for single digit character readings
            else:
                numbers = '{0},'.format(resultList[0])
                f.write(numbers)
        f.close()
        ### NOTE - THE FORMULAS ONLY ACCOUNT FOR SINGLE OR TRIPLE CHARACTERS--NO DOUBLE OR LONGER ###

    """
    This function returns the name and number of columns of a particular template
    @param template_name: name of the template being processed
    @return the number of rows in a column and the name the crop should be saved as
    """
    def getFileName(self, template_name):
        if template_name == 'pssdate.jpg':
            col_num = 0
            save_file_name = 'ssdate'
        if template_name == 'psssvirb1.jpg':
            col_num = 0
            save_file_name = 'svirb1'
        if template_name == 'psssvirb2.jpg':
            col_num = 0
            save_file_name = 'svirb2'
        if template_name == 'pssmcrtrb1.jpg':
            col_num = 0
            save_file_name = 'mcrtrb1'
        if template_name == 'pssmcrtrb2.jpg':
            col_num = 0
            save_file_name = 'mcrtrb2'
        if template_name == 'pssrb1p4pH.jpg':
            col_num = 0
            save_file_name = 'rb1p4pH'
        if template_name == 'pssrb1p4mlss.jpg':
            col_num = 0
            save_file_name = 'rb1p4mlss'
        if template_name == 'pssrb1p4min.jpg':
            col_num = 0
            save_file_name = 'rb1p4min'

        if template_name == 'pssrb2p4pH.jpg':
            col_num = 0
            save_file_name = 'rb2p4pH'
        if template_name == 'pssrb2p4pH.jpg':
            col_num = 0
            save_file_name = 'rb2p4pH'
        if template_name == 'pssrb2p4mlss.jpg':
            col_num = 0
            save_file_name = 'rb2p4mlss'
        if template_name == 'pssrb2p4min.jpg':
            col_num = 0
            save_file_name = 'rb2p4min'
        if template_name == 'pssrasmlss.jpg':
            col_num = 0
            save_file_name = 'rasmlss'
        if template_name == 'pssrb1p2DO.jpg':
            col_num = 0
            save_file_name = 'rb1p2DO'
        if template_name == 'pssrb1p3.jpg':
            col_num = 0
            save_file_name = 'rb1p3'
        if template_name == 'pssrb2p2DO.jpg':
            col_num = 0
            save_file_name = 'rb2p2DO'
        if template_name == 'pssrb2p3.jpg':
            col_num = 0
            save_file_name = 'rb2p3'

        if template_name == 'psslimesilo.jpg':
            col_num = 0
            save_file_name = 'limesilo'
        if template_name == 'pssBDBpolygas.jpg':
            col_num = 0
            save_file_name = 'BDBpolygas'
        if template_name == 'pssairscrubber.jpg':
            col_num = 0
            save_file_name = 'airscrubber'
        if template_name == 'psscarbontank1.jpg':
            col_num = 0
            save_file_name = 'cartank1'
        if template_name == 'psscarbontank2.jpg':
            col_num = 0
            save_file_name = 'cartank1'
        if template_name == 'psshypotank1.jpg':
            col_num = 0
            save_file_name = 'hypotank1'
        if template_name == 'psshypotank2.jpg':
            col_num = 0
            save_file_name = 'hypotank2'
        if template_name == 'pssbisulfatetotals.jpg':
            col_num = 0
            save_file_name = 'bisulfatetotals'
        if template_name == 'psscaustictank1.jpg':
            col_num = 0
            save_file_name = 'caustank1'
        if template_name == 'psscaustictank2.jpg':
            col_num = 0
            save_file_name = 'caustank2'
        if template_name == 'pssalumtank1.jpg':
            col_num = 0
            save_file_name = 'alumtank1'
        if template_name == 'pssalumtank2.jpg':
            col_num = 0
            save_file_name = 'alumtank2'

        if template_name == 'pssinfluentpH.jpg':
            col_num = 2
            save_file_name = 'influentpH'
        if template_name == 'pssinfluenttemp.jpg':
            col_num = 2
            save_file_name = 'influenttemp'
        if template_name == 'pssphosphorousPO4.jpg':
            col_num = 2
            save_file_name = 'phosphorousPO4'
        if template_name == 'pssammonianitro.jpg':
            col_num = 2
            save_file_name = 'ammnitro'
        if template_name == 'pssCCDO.jpg':
            col_num = 2
            save_file_name = 'CCDO'
        if template_name == 'pssCCpH.jpg':
            col_num = 2
            save_file_name = 'CCpH'
        if template_name == 'psstotalCL2.jpg':
            col_num = 2
            save_file_name = 'totalCL2'
        if template_name == 'pssCL2dose.jpg':
            col_num = 2
            save_file_name = 'CL2dose'
        if template_name == 'pssbisulfitedose.jpg':
            col_num = 2
            save_file_name = 'bisulfatedose'
        if template_name == 'pssflow.jpg':
            col_num = 2
            save_file_name = 'flow'
        if template_name == 'pssinit.jpg':
            col_num = 2
            save_file_name = 'initials'

        if template_name == 'pssAVinfluentpH.jpg':
            col_num = 0
            save_file_name = 'avinflupH'
        if template_name == 'pssAVinfluenttemp.jpg':
            col_num = 0
            save_file_name = 'avinflutemp'
        if template_name == 'pssAVCCammnitro.jpg':
            col_num = 0
            save_file_name = 'avCCammnitro'
        if template_name == 'pssAVCCDO.jpg':
            col_num = 0
            save_file_name = 'avCCDO'
        if template_name == 'pssAVCCpH.jpg':
            col_num = 0
            save_file_name = 'avCCpH'
        if template_name == 'pssAVCCtotalCL2.jpg':
            col_num = 0
            save_file_name = 'avCCtotCL2'
        if template_name == 'pssemplinit.jpg':
            col_num = 0
            save_file_name = 'emplinit'

        return save_file_name, col_num

    """
    This function clears the pre-existing file to replace it with new information
    @param text_file: the output file
    """
    def erase_file_contents(self, text_file):
        open(text_file, 'w').close()

import cv2
import imutils
from PIL import Image
import numpy as np


class MatchTemplate:
    def __init__(self):
        pass

    """
    This function uses OpenCVs templatematch function to find a column on a form.
    @param form: the form to be read
    @param columnToFind: an individual column heade
    @param columnHead: an entire row of column headers
    @param form_type: the form type selected by user
    NOTE: code taken from https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    """
    def template_match(self, form, columnToFind, columnHead, form_type):
        # load the image, convert it to grayscale, and detect edges
        template = cv2.imread(columnHead)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        # cv2.imshow("Header Template", template)

        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        image = cv2.imread(form)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        file_name, col_num = SampleSheet.getFileName(columnToFind)

        # NOTE: should maybe turn getFileName into 2 functions, one to return
        # the filename and one to get # of columns
        size_change = endY - startY
        new_endy = endY + (size_change * (col_num + 1))

        # print(startY, new_endy, startX, endX)
        roi = image[startY:new_endy, startX:endX]
        cv2.imwrite("chopped.jpg", roi)

        # finds the template inside the form
        template = cv2.imread(columnToFind)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        # cv2.imshow("Template", template)

        image = cv2.imread("chopped.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None

        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        img = cv2.resize(image, (960, 540))
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

        self.crop_function(self, form, startX, startY, endX, endY, columnToFind)

    """
    This function runs down the column of a template's coordinates and crops each box
    @param form: the form to be read
    @param coordinates: the coordinates of a located column on a form
    @param template_name: the name of the column that has been found
    """
    def crop_function(self, form, x0, y0, x1, y1, template_name):
        save_file_name, col_num = SampleSheet.getFileName(template_name)
        size_chg = y1 - y0
        # establish a counter for rows
        row = 1
        # create a baseline variable to reset the save_file_name variable
        save_name = save_file_name
        # add column number to end of save_file_name
        save_file_name += '{0}'.format(row)
        # add the file ending to end of save_file_name
        save_file_name += '.jpg'

        # crop out the first box from the form
        self.crop(form, (x0, (y0 + size_chg), x1, (y1 + size_chg)), save_file_name)

        # reset save_file_name
        save_file_name = save_name

        row += 1
        # create baseline variables for coordinate calculation
        newy0, newy1 = self.getCoords(y0, y1, size_chg)

        for col in range(col_num):
            newy0, newy1 = self.getCoords(newy0, newy1, size_chg)
            save_file_name += '{0}'.format(row)
            save_file_name += '.jpg'
            self.crop(form, (x0, newy0, x1, newy1), save_file_name)
            save_file_name = save_name
            row += 1

    """    
    This function receives coordinates of item to be cropped and returns the next row's y coordinates
    @param y0: the start of y
    @param y1: the end of y
    @param size_chg: the difference between y1 - y0
    """
    def getCoords(self, y0, y1, size_chg):
        y0 = y1
        y1 += size_chg
        return y0, y1

    """    
    This function crops a box from the form in a specified column
    @param image_path: the form being processed
    @param coords: coordinates where a box in a specific column has been located
    @param saved_location: the name and filetype that the box should be saved as 
    """
    def crop(self, image_path, coords, saved_location):
        """
        @param image_path: The path to the image to edit
        @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
        @param saved_location: Path to save the cropped image
        """
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)

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

    """
    This function prepares the image to be processed by correcting size, shape, and color of each temporary PNG image file
    @param image: a specific box from the form
    @return the prepared image in numpy array form
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
    @param image: a specific box from the form
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