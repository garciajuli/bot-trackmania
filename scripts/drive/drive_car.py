from os.path import isdir
from os import mkdir
from time import sleep
from ..utils.screenshot_data import ScreenData
from cv2 import namedWindow, resizeWindow, getTickCount, getTickFrequency
from cv2 import resize, cvtColor, line, putText, circle, imshow, waitKey, imwrite
from cv2 import COLOR_BGR2RGB, WINDOW_NORMAL, FONT_HERSHEY_COMPLEX_SMALL, LINE_AA
from PIL.ImageGrab import grab
from tensorflow.keras.models import load_model
from tensorflow import data, expand_dims
from pynput import keyboard
import pyvjoy
import numpy as np
from joblib import load

class Drive:

    def __init__(self, modelName, modelType, showRace, saveName=None):
        self.modelPath = './model/drive/'
        self.savePathRoot = './raceSave/'

        self.saveName = saveName
        self.showRace = showRace
        self.modelType = modelType

        if self.saveName is not None:
            self.videoRace = []
            if not isdir(self.savePathRoot + self.saveName):
                mkdir(self.saveName)

        if(self.modelType == "joblib"):
            self.model = load(self.modelPath + modelName)
        else:
            self.model = load_model(self.modelPath + modelName)

        ## init joystick
        self.joystick = pyvjoy.VJoyDevice(1)
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
        self.joystick.set_axis(pyvjoy.HID_USAGE_RZ, 0x1)
        self.joystick.set_axis(pyvjoy.HID_USAGE_SL0, 0x1)

        ## Init screen data
        self.dataImage = ScreenData()
        self.run = True
        self.displayInfo = True

    def start(self):
        if self.showRace:
            namedWindow("Race", WINDOW_NORMAL)
            resizeWindow("Race", 960,540)

        answer = input("Press O to start driving (you get 3 seconds to get focus on the game window) also press another key to cancel.")
        if answer == "O":
            self.keyboardListener()
            sleep(3)

            while self.run:
                tickmark = getTickCount()

                imageOrigin = self.getScreen()
                image = imageOrigin.copy()

                self.dataImage.setImage(image)

                if(self.modelType == "joblib"):
                    speed, distances, pos = self.dataImage.getInfos()

                    # format data for model
                    distancesFormat = []
                    # speedFormat = [round((speed/500), 3)]

                    for i, distance in enumerate(distances):
                        distance = round((distance / 700), 3)
                        distancesFormat.append(distance)
                        if self.showRace and self.displayInfo:
                            circle(image, (pos[i][0], pos[i][1]), 2, (0,0,255), -1)
                            putText(image, str(distance), (pos[i][0]-20, pos[i][1]-20), FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1, LINE_AA)
                            line(image , (pos[i][0], pos[i][1]), (480, 540), (0, 255, 0), 2)

                    # inputs = distancesFormat + speedFormat
                    # inputs = expand_dims(inputs, 0)
                    # inputs_dataset = data.Dataset.from_tensor_slices(inputs).batch(1)
                    # predic = self.model.predict(inputs_dataset)[0]

                    inputs = distances + [speed]
                    predic = self.model.predict([inputs])[0]

                    fps=getTickFrequency()/(getTickCount()-tickmark)
                    if self.showRace:
                        putText(image, "FPS: {:05.2f}, speed: {:d}, press TAB to Display infos".format(fps, speed), (10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        imshow("Race",image)
                        waitKey(1)

                    if self.saveName is not None:
                        self.videoRace.append(image)

                else:

                    imageOrigin2 = imageOrigin[np.newaxis, :]

                    predic = self.model.predict(imageOrigin2)[0]
                    # predic = [p / 100000000000000 for p in predic]
                    # predic = np.around(predic,2)
                    # print("predic", predic)

                    fps=getTickFrequency()/(getTickCount()-tickmark)
                    if self.showRace:
                        putText(image, "FPS: {:05.2f}, press TAB to Display infos".format(fps), (10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        imshow("Race",image)
                        waitKey(1)

                self.controllJoystock(predic)

    def getScreen(self):
        screen = grab()
        image = cvtColor(np.array(screen), COLOR_BGR2RGB)

        # calculate the 50 percent of original dimensions
        width = int(image.shape[1] * 50 / 100)
        height = int(image.shape[0] * 50 / 100)

        imageScale = resize(image, (width, height))

        return imageScale

    def controllJoystock(self, predict):
        print("predict : ", predict)
        MAX_VJOY = 32767
        MIDDLE_VJOY = MAX_VJOY//2

        dataAxeX = int(predict[0] * MIDDLE_VJOY + MIDDLE_VJOY)
        self.joystick.set_axis(pyvjoy.HID_USAGE_X, dataAxeX)

        if predict[1] > 0.5:
            self.joystick.set_axis(pyvjoy.HID_USAGE_RZ, 0x8000)
        else:
            self.joystick.set_axis(pyvjoy.HID_USAGE_RZ, 0x1)

        if predict[2] > 0.5:
            self.joystick.set_axis(pyvjoy.HID_USAGE_SL0, 0x8000)
        else:
            self.joystick.set_axis(pyvjoy.HID_USAGE_SL0, 0x1)

    def keyboardListener(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.listener.stop()
                self.run = False
            if key == keyboard.Key.tab:
                self.displayInfo = self.displayInfo == False
            if  key == keyboard.Key.backspace:
                self.joystick.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
                self.joystick.set_axis(pyvjoy.HID_USAGE_RZ, 0x1)
                self.joystick.set_axis(pyvjoy.HID_USAGE_SL0, 0x1)
        except KeyError:
            pass


    def on_release(self, key):
        try:
            if key == keyboard.Key.esc:
                print('End record input')
        except KeyError:
            pass