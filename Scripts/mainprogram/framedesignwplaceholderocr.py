"""
This is the testing file for the GUI of the Waverunners OCR project. It will run all the way from the first screen to
the last, and loop back to the beginning again. Instead of processing a user-uploaded form, this program replaces the
backend logic with a 'placeholderocr' function that simply writes a bunch of numbers to a file, then moves to the next
screen.
"""
from os.path import sep, expanduser, isdir, dirname
from kivy.config import Config
import threading

Config.set('kivy', 'exit_on_escape', 1)
Config.set('input', 'mouse', 'mouse, disable_multitouch')
# Config.set('graphics', 'fullscreen', 'auto')

import kivymd.theming

import shutil
import csv
import pandas as pd

import tkinter as tk
#from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, StringProperty
from PIL import *
from PIL import Image
# from biosolids import BioSolids
# from npdes import NPDES
# from samplesheet import SampleSheet
from userdatabase import Userdatabase
from placeholderOCR import countTo1000
import matplotlib.pyplot as plt
import numpy as np


# this is the login screen
class LoginWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def loginBtn(self):
        self._shadow = App.get_running_app().theme_cls.quad_shadow

        #this function validates the username and password for login feature
        #this was built with LDAP in mind--use the self.email.text to assign info to email TextInput box
        if db.validate(self.email.text, self.password.text):
            MainWindow.current = self.email.text
            self.reset()
            self.manager.current = "welcome"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        self.manager.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""

    def show_password(self, field, button):  # currently inoperational
        """
         Called when you press the right button in the password field
         for the screen TextFields.

         instance_field: kivy.uix.textinput.TextInput;
         instance_button: kivymd.button.MDIconButton;

         """
        # Show or hide text of password, set focus field
        # and set icon of right button.
        field.password = not field.password
        field.focus = True
        button.icon = "eye" if button.icon == "eye-off" else "eye-off"


# this class won't be necessary in the future, so delete this AND the backend info in design.kv
class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count(
                "@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)

                self.reset()
                # calls the popup to confirm user is successfully created
                Submit_Success()
                self.manager.current = "login"
            else:
                invalidForm()
        else:
            invalidForm()

    def login(self):
        self.reset()
        self.manager.current = "login"

    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""


# this screen displays a welcome message and offers a JPG to be uploaded
class MainWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.menu_items = [
            {
                "viewclass": "MDMenuItem",
                "text": "Biosolids Log",
                "callback": self.callback_for_menu_items,
            },
            {
                "viewclass": "MDMenuItem",
                "text": "Sample Sheet Log",
                "callback": self.callback_for_menu_items,
            },
            {
                "viewclass": "MDMenuItem",
                "text": "NPDES Log",
                "callback": self.callback_for_menu_items,
            }
        ]

    def callback_for_menu_items(self, *args):
        pass

    def change_variable(self, value):
        global form_type
        form_type = value
        #call popup to verify user selection
        File_Selected(form_type)
        ###NOTE: need to figure out how to return the value variable elsewhere in the program!###
        ###We'll need to use it to tell the program which form it'll be processing later on.###
        return form_type

    def changeScreen(self):
        self.manager.current = 'jpguploaded'


# this screen will hold the loading animation
class LoadingPage(Screen):
    def on_enter(self):
        self.threading_function()

    def threading_function(self):
        mythread = threading.Thread(target=self.call_ocr_function)
        mythread.start()

    def call_ocr_function(self):
        print(form_type)
        global output
        # if form_type == 'Sample Sheet Log':
        #     output = SampleSheet.startProcessing(SampleSheet, image_file_directory, form_type)
        # if form_type == 'NPDES Log':
        #     output = NPDES.startProcessing(NPDES, image_file_directory, form_type)
        # if form_type == 'Biosolids Log':
        #     output = BioSolids.startProcessing(BioSolids, image_file_directory, form_type)

        ### the below line will run the program without utilizing the external files
        ### use the below line for testing
        output = countTo1000.placeholder_ocr(countTo1000)

        self.manager.current = 'ocrcomplete'


# this screen will show the uploaded JPG image on screen and will offer chance to
# upload a different file or convert to OCR
class JPGIsUploaded(Screen):
    def on_pre_enter(self):
        # assigns user-selected image from open_file function to the Image widget
        self.ids.image.source = image_file_directory

    #### bug here ####
    def clear_image(self):
        self.ids.image.source = ""

    ### picture does not update ###
    def assign_image(self):
        self.ids.image.source = image_file_directory


# this screen displays the converted OCR data and offers options to either edit
# data or submit the data
class OCRComplete(Screen):
    def on_enter(self):
        self.read_file()
        self.ids.iimage.source = image_file_directory

    ### this function is supposed to read the csv file onto the GUI screen###
    def read_file(self):
        file = open("Results.csv", "r")
        #csvtext = file.read()
        #self.ids.input.text = pd.read_csv(file)
        # print(file.read())
        self.ids.input.text = file.read()

    # this function will open a new window to allow the user to edit the file
    def edit_file(self):
        pass


# this page contains an FAQ for the program
class HelpPage(Screen):
    pass


# this is the screen manager for all the screens.
class WindowManager(ScreenManager):
    pass


# displays popup for invalid username/passwords
def invalidLogin():
    pop = Popup(title='Invalid Login',
                content=Label(text='Invalid username or password.'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# displays popup for invalid form
def invalidForm():
    pop = Popup(title='Invalid Form',
                content=Label(text='Please fill in all inputs with valid information.'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# displays when a new user is successfully created
def Submit_Success():
    pop = Popup(title='Submitted',
                content=Label(text='Successful!'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# will display when file is successfully saved
def File_Saved():
    pop = Popup(title='File Successfully Saved',
                content=Label(text='Successful!'),
                size_hint=(None, None), size=(400,400))
    pop.open()


def File_Selected(value):
    pop = Popup(title='Form Selection',
                content=Label(text='You selected ' + str(value)),
                size_hint=(None, None), size=(400,400))
    pop.open()


db = Userdatabase("users.txt")
kv = Builder.load_file("design.kv")


# this class is what builds and runs the application
class FrameDesign(App):
    #theme_cls sets color scheme
    theme_cls = kivymd.theming.ThemeManager()
    title = "Parkway OCR Application"

    def build(self):
        self.theme_cls.primary_palette = 'Blue'
        #this is returning the GUI structure, the ScreenManager
        wm = WindowManager()
        return wm

    # this function opens a file manager for the user to select a JPG file
    ### tkinter - change to Kivy ###
    def open_file(self):
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        global image_file_directory

        image_file_directory = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        root.destroy()
        return image_file_directory

    # this function opens a file manager for the user to select a JPG file
    ### tkinter - change to Kivy ###
    def save_file(self):
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        f = filedialog.asksaveasfilename(defaultextension=".csv")

        shutil.copy('Results.csv', f)

        File_Saved()

        root.destroy()

    # displays a box asking user if they want to leave application
    ### tkinter - change to kivy ###
    def exit_app(self):
        # this creates a TK canvas window. Is there a way to connect the 'root' variable
        # back into Kivy? self.manager doesn't work
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application?',
                                           icon='warning')
        if MsgBox == 'yes':
            root.destroy()
            quit()
        else:
            tk.messagebox.showinfo('Return', 'You will now return to the application screen.')
            root.destroy()


if __name__ == '__main__':
    FrameDesign().run()
