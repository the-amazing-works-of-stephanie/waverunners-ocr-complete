import cv2
import numpy as np
import imutils
from PIL import Image
from matplotlib import pyplot as plt


# code taken from https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
# parameters are: form (the form to be read), columnToFind (individual column headers), and
# columnHead (the entire row of column headers)
def template_match(form, columnToFind, columnHead):
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(columnHead)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Header Template", template)

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

    file_name, col_num = getFileName(columnToFind)
    # NOTE: should maybe turn getFileName into 2 functions, one to return
    # the filename and one to get # of columns
    size_change = endY - startY
    new_endy = endY + (size_change * (col_num+1))

    #print(startY, new_endy, startX, endX)
    roi = image[startY:new_endy, startX:endX]
    cv2.imwrite("chopped.jpg", roi)

    # finds the template inside the form
    template = cv2.imread(columnToFind)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Template", template)

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
    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # img = cv2.resize(image, (960, 540))
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

    return startX, startY, endX, endY


# crops a box out
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


# runs down the column of a template's coordinates and crops each box
def crop_function(form, x0, y0, x1, y1, template_name):
    save_file_name, col_num = getFileName(template_name)
    size_chg = y1 - y0
    # establish a counter for columns
    column = 1
    # create a baseline variable to reset the save_file_name variable
    save_name = save_file_name
    # add column number to end of save_file_name
    save_file_name += '{0}'.format(column)
    # add the file ending to end of save_file_name
    save_file_name += '.jpg'

    # crop out the first box from the form
    crop(form, (x0, (y0 + size_chg), x1, (y1 + size_chg)), save_file_name)

    # reset save_file_name
    save_file_name = save_name

    column += 1
    # create baseline variables for coordinate calculation
    newy0, newy1 = getCoords(y0, y1, size_chg)

    for col in range(col_num):
        newy0, newy1 = getCoords(newy0, newy1, size_chg)
        save_file_name += '{0}'.format(column)
        save_file_name += '.jpg'
        crop(form, (x0, newy0, x1, newy1), save_file_name)
        save_file_name = save_name
        column += 1


# receives coordinates of item to be cropped and returns the next row's y coordinates
def getCoords(y0, y1, size_chg):
    y0 = y1
    y1 += size_chg
    return y0, y1


# returns the name and number of columns of a particular template
def getFileName(template_name):
    ###Biosolids log templates start here###
    if template_name == "BioSolDate.jpg":
        num_of_rows = 7
        save_file_name = 'InitDate'
    if template_name == 'BioSolInitTime.jpg':
        num_of_rows = 7
        save_file_name = 'InitTime'
    if template_name == 'BioSolInitpH.jpg':
        num_of_rows = 7
        save_file_name = 'InitpH'
    if template_name == 'BioSolInitTempC.jpg':
        num_of_rows = 7
        save_file_name = 'InitTempC'
    if template_name == 'BioSolInitAdjpH.jpg':
        num_of_rows = 7
        save_file_name = 'InitAdjpH'
    if template_name == 'BioSolInitInitials.jpg':
        num_of_rows = 7
        save_file_name = 'InitInitials'

    if template_name == 'BioSol2HrTime.jpg':
        num_of_rows = 7
        save_file_name = '2HrTime'
    if template_name == 'BioSol2HrpH.jpg':
        num_of_rows = 7
        save_file_name = '2HrpH'
    if template_name == 'BioSol2HrTempC.jpg':
        num_of_rows = 7
        save_file_name = '2HrTempC'
    if template_name == 'BioSol2HrAdjpH.jpg':
        num_of_rows = 7
        save_file_name = '2HrAdjpH'
    if template_name == 'BioSol2HrInitials.jpg':
        num_of_rows = 7
        save_file_name = '2HrInitials'

    if template_name == 'BioSol24HrTime.jpg':
        num_of_rows = 7
        save_file_name = '24HrTime'
    if template_name == 'BioSol24HrpH.jpg':
        num_of_rows = 7
        save_file_name = '24HrpH'
    if template_name == 'BioSol24HrTempC.jpg':
        num_of_rows = 7
        save_file_name = '24HrTempC'
    if template_name == 'BioSol24HrAdjpH.jpg':
        num_of_rows = 7
        save_file_name = '24HrAdjpH'
    if template_name == 'BioSol24HrInitials.jpg':
        num_of_rows = 7
        save_file_name = '24HrInitials'

    if template_name == 'BioSolInitTrailer.jpg':
        num_of_rows = 7
        save_file_name = 'BioTrailer'

    if template_name == 'BioSolLimeBuffer7-1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeBuffer71'
    if template_name == 'BioSolLimeBuffer7-2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeBuffer72'
    if template_name == 'BioSolLimeBuffer10-1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeBuffer101'
    if template_name == 'BioSolLimeBuffer10-2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeBuffer102'
    if template_name == 'BioSolLimeBuffer1245-1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeBuffer12451'
    if template_name == 'BioSolLimeBuffer1245-2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeBuffer12452'
    if template_name == 'BioSolLimeBufferTempC-1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeTempC1'
    if template_name == 'BioSolLimeBufferTempC-2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeTempC2'
    if template_name == 'BioSolLimeDate1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeDate1'
    if template_name == 'BioSolLimeDate2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeDate2'
    if template_name == 'BioSolLimeInitials1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeInitials1'
    if template_name == 'BioSolLimeInitials2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeInitials2'
    if template_name == 'BioSolLimeTime1.jpg':
        num_of_rows = 1
        save_file_name = 'LimeTime1'
    if template_name == 'BioSolLimeTime2.jpg':
        num_of_rows = 0
        save_file_name = 'LimeTime2'

    ###Sample Sheet templates start here###
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

    ###NPDES templates start here###
    if template_name == "NpdesDate.jpg":
        num_of_rows = 0
        save_file_name = 'NpdesDateProcessed'
    if template_name == 'NpdesTimeCollected.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesTimeCollectedProcessed'
    if template_name == 'NpdesTimeAnalyzed.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesTimeAnalyzedProcessed'
    if template_name == 'NpdesFinalDo.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesFinalDoProcessed'
    if template_name == 'NpdesBarometricPressure.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesBarometricPressureProcessed'
    if template_name == 'NpdesCalibDo.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesCalibDoProcessed'
    if template_name == 'NpdesAirTemp.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesAirTempProcessed'
    if template_name == 'NpdesFinalPh.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesFinalPhProcessed'
    if template_name == 'NpdesPhSlope.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesPhSlopeProcessed'

    if template_name == 'NpdesDateOnPhProbe.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesDateOnPhProbeProcessed'
    if template_name == 'NpdesBuffer7Probe.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesBuffer7ProbeProcessed'

    if template_name == 'NpdesBuffer4Probe.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesBuffer4ProbeProcessed'
    if template_name == 'NpdesBuffer10Probe.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesBuffer10ProbeProcessed'
    if template_name == 'NpdesFinalCl2.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesFinalCl2Processed'
    if template_name == 'NpdesFlowMgd.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesFlowMgdProcessed'
    if template_name == 'NpdesEmployeeSignature.jpg':
        num_of_rows = 2
        save_file_name = 'NpdesEmployeeSignatureProcessed'

    if template_name == 'NpdesLabDate.jpg':
        num_of_rows = 0
        save_file_name = 'NpdesLabDateProcessed'

    if template_name == 'NpdesLabTempC.jpg':
        num_of_rows = 0
        save_file_name = 'NpdesLabTempCProcessed'
    if template_name == 'NpdesLabInt.jpg':
        num_of_rows = 0
        save_file_name = 'NpdesLabIntProcessed'
    if template_name == 'NpdesRainfall.jpg':
        num_of_rows = 0
        save_file_name = 'NpdesRainfallProcessed'
    if template_name == 'NpdesAverageFinalDo.jpg':
        num_of_rows = 0
        save_file_name = 'NpdesAverageFinalDoProcessed'

    return save_file_name, col_num


# this is hardcoded in for now, but in the real program, 'form' is a global variable
form = '3263_001-1.jpg'

x0, y0, x1, y1 = template_match(form, 'BioSolDate.jpg', 'headercol1-0.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolDate.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitTime.jpg', 'headercol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitpH.jpg', 'headercol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitTempC.jpg', 'headercol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitAdjpH.jpg', 'headercol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitAdjpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitInitials.jpg', 'headercol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, 'BioSolInitInitials.jpg')

x0, y0, x1, y1 = template_match(form, 'BioSol2HrTime.jpg', 'headerCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol2HrTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrTempC.jpg', 'headerCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol2HrTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrAdjpH.jpg', 'headerCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol2HrAdjpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrInitials.jpg', 'headerCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol2HrInitials.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrpH.jpg', 'headerCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol2HrpH.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSol24HrTime.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol24HrTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrpH.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol24HrpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrTempC.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol24HrTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrAdjpH.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol24HrAdjpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrInitials.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSol24HrInitials.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolInitTrailer.jpg', 'headerCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolInitTrailer.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer7-1.jpg', 'headerCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer7-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer10-1.jpg', 'headerCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer10-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer1245-1.jpg', 'headerCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer1245-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBufferTempC-1.jpg', 'headerCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBufferTempC-1.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeDate1.jpg', 'headerCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeDate1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeTime1.jpg', 'headerCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeTime1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeInitials1.jpg', 'headerCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeInitials1.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer7-2.jpg', 'headerCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer7-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer10-2.jpg', 'headerCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer10-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer1245-2.jpg', 'headerCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBuffer1245-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBufferTempC-2.jpg', 'headerCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeBufferTempC-2.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeDate2.jpg', 'headerCol3-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeDate2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeTime2.jpg', 'headerCol3-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeTime2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeInitials2.jpg', 'headerCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolLimeInitials2.jpg")
