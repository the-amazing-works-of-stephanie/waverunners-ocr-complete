# waverunners-ocr-complete

Hello, and welcome to the Waverunners OCR intern project! 

We worked on three major parts during this project, and so the project has been split up into three folders: 
1) keras -- this is where we trained a neural network on the MNIST dataset, saved the model, and did some preliminary testing on contouring and writing predicted numbers to a file.
2) mainprogram -- contains all the parts of the program: the GUI/front end logic and all backend logic (file processing, to include template matching, contouring, and predicting file contents using model).
3) templatematchingtest -- this is where we tested OpenCV's 'templatematch' function. 

*Please note, this program was designed as 3 separate programs, not one. Therefore, if you'd like to run any of the three above pieces, please copy the contents of the desired folder into a new PyCharm project. 

###### PROGRAM PURPOSE: 
Parkway Water Resource Recovery Facility has to manually record and upload a few sets of data every single day. This manual data entry takes a lot of time that could be spent elsewhere, so we decided to design a smart program that would intercept the handwritten form, locate columns (for data output), break them down into individual numbers, then read and record them via a machine learning model that we trained to read digits. 

Here are some in-depth details about the project: 

###### MAIN PROGRAM: 
- These are the COMPLETE program files. 
- The main program itself is held in the 'framework' or 'frameworkwplaceholderocr' folders. The GUI was designed with Kivy and KivyMD, both of which allow an HTML-esque '.kv' (aka design.kv) to act as the design's backend programming. The processing for each form type (Biosolids, NPDES, and SampleSheet), to include code and templates, are contained within separate folders inside the mainprogram folder. 
- The logic for this program is complicated, and so has been turned into a diagram to show all the connecting pieces. It is uploaded in PDF form onto the ShareDrive. (NOTE TO EDITOR: Upload here instead)
- Since the majority of WSSC developers utilize Java, we have done our best to make this program look as close to Java as possible. There are Javadoc comments for most of the functions, and the majority of the code is reusable (i.e. the contours.py and templatematch.py are used by all three forms, and is not customized). (NOTE TO EDITOR: fix this)
- Contours.py and templatematch.py are NOT the latest versions of the functions. Full Javadoc comments are in samplesheet.py/biosolids.py/npdes.py.
- HOWEVER, we could not figure out how to connect the form-related python files (samplesheet.py, biosolids.py, and npdes.py) to call on contours.py or templatematch.py. There seems to be an issue with double crossed headers (i.e. samplesheet.py calls templatematch.py, but templatematch.py needs a function in samplesheet.py so the two classes are pingponging back and forth.) To remedy this issue, we just copy/pasted the contents of contour.py and templatematch.py into each file. 

###### KERAS: 
- This section contains all information related to training a model on the MNIST dataset and contour testing.

Here is an explanation of each file: 
- keras-mnist-test: this is an MNIST trainer taken directly from Keras' website
- danielKerasClassifier: this was an MNIST trainer made by our summer 2019 intern, Daniel Johnson. It was used to train 'Keras_model4.h5'
- handsOnMachine Learning: this is an MNIST trainer taken directly from the textbook, ____. We used it to train 'stephanieTest1_11_27_19.h5' and 'stephanieTest2.h5' (NOTE TO EDITOR: textbook name is missing)
- sitePointTraining: this is an MNIST trainer taken from this website (https://www.sitepoint.com/keras-digit-recognition-tutorial/). We used it to train 'stephanieTest3.h5', and this is the model we've been utilizing for real tests because it has about 95% accuracy. 

- handsonML_loadmodel: this is a tutorial taken from the textbook, ____, to check its accuracy against given numbers. We fed it images taken directly from the MNIST dataset, which you can find inside the Templates folder. (NOTE TO EDITOR: textbook name is missing)
- daniel_multinumber_model_tester: this code was originally written by our summer 2019 intern, Daniel Johnson. It takes a model and a form, cuts out rows from a column, and reads the digits one by one to write them into a file. 
- image_preprocessing_test: this was my attempt to test the function 'model.predict'. It didn't turn out well; the file doesn't work. I'm leaving it in if you'd like to experiment on your own.

###### TEMPLATEMATCHINGTEST:
- This section will allow you to test out the functionality of opencv's templatematch function and check the output, without having to run through the GUI every time. We used it to test accuracy of crops. 
- Most of the functions should look familiar if you've perused the mainprogram files. If you want to know details on what each of these functions do, please see samplesheet.py, biosolids.py, or npdes.py. 
- Currently (12/31/2019) the program is set up to run through the Biosolids form. If you want to run through a different one, please use the code from the npdes.py or samplesheet.py files. Be aware that this file runs a bit differently from mainprogram. The function calls should look like this: 
1) x0, y0, x1, y1 = template_match(form, 'BioSolDate.jpg', 'headercol1-0.jpg') 	//saving the coordinates of a located column header
2) crop_function("chopped.jpg", x0, y0, x1, y1, "BioSolDate.jpg")			//using those coordinates to crop the rows beneath the indicated column

*NOTE: running through template_matching.py takes a few minutes. While the program is running, you should see files spawning in the source folder as the program locates and crops boxes from the form. Once the program returns 0, it has finished cropping. 

###### LIBRARIES NEEDED FOR THIS PROJECT: 
- Python 3.6 or 3.7 (Kivy and KivyMD do not currently (12/2019) support Python 3.8)
- Kivy (latest version should be OK, but we built on 1.11.1)
- KivyMD (latest version should be OK, but we built on 0.102.0)
- Keras (latest version should be OK, but we built on 2.3.1)
- Pillow (latest version should be OK, but we built on 6.2.1)
- numpy (latest version should be OK, but we built on 1.18.0)
- matplotlib (latest version should be OK)
- pandas (latest version should be OK, but we built on 0.25.3)
- tensorflow (version 1.14.0) -- this could be upgraded to 2.0.0, but at the time of writing utilizing Tensorflow GPU wasn't available on 2.0.0
- opencv (version 3.4.2.17) -- there were some API-breaking changes after this version, so unless you were to locate them, you wouldn't be able to use another version
