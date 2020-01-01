import cv2
import imutils
from PIL import Image
import numpy as np
from module_biosolids.biosolids import BioSolids
from module_npdes.npdes import NPDES
from module_samplesheet.samplesheet import SampleSheet


class MatchTemplate:
    def __init__(self):
        pass

    """
    This function *******
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

        if form_type == 'npdes':
            file_name, col_num = NPDES.getFileName(columnToFind)
        if form_type == 'biosolids':
            file_name, col_num = BioSolids.getFileName(columnToFind)

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

        self.crop_function(self, form, startX, startY, endX, endY, columnToFind, form_type)

        #return startX, startY, endX, endY

    """This function runs down the column of a template's coordinates and crops each box
    @param form: 
    @param coordinates: 
    @param template_name: 
    """
    def crop_function(self, form, x0, y0, x1, y1, template_name, form_type):
        if form_type == 'npdes':
            save_file_name, col_num = NPDES.getFileName(template_name)
        if form_type == 'biosolids':
            save_file_name, col_num = BioSolids.getFileName(template_name)
        if form_type == 'samplesheet':
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

    """This function receives coordinates of item to be cropped and returns the next row's y coordinates
    @param y0: 
    @param y1: 
    @param size_chg: 
    """
    def getCoords(self, y0, y1, size_chg):
        y0 = y1
        y1 += size_chg
        return y0, y1

    """This function ***********
    @param image_path: 
    @param coords: 
    @param saved_location: 
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