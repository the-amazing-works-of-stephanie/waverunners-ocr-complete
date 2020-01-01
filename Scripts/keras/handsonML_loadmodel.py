# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
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

# load an image and predict the class
def run_example():
    model = load_model('stephanieTest1_11_27_19.h5')
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    img = load_image('00.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('11.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('22.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('33.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('44.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('55.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('66.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('77.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('88.png')
    digit = model.predict_classes(img)
    print(digit[0])
    img = load_image('99.png')
    digit = model.predict_classes(img)
    print(digit[0])

# entry point, run the example
run_example()
