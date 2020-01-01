# this class will act as a placeholder for the OCR--it will run a function to count to 1000,
# save those numbers into a text file, and return that text file back to framedesign.py.


class countTo1000:
    def __init__(self):
        pass

    def placeholder_ocr(self):
        counter = 0
        return_value = False

        global ocr_file

        ocr_file = open("ocr.txt", "w+")

        while counter != 9999999:
            if counter < 9999999:
                counter += 1
            ocr_file.write(str(counter))
            if counter == 9999999:
                number = 1
                ocr_file.close()

        return return_value