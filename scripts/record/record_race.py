from os.path import exists, isdir
from os import mkdir
from json import dump, load
from time import sleep, time
from cv2 import resize, cvtColor, COLOR_BGR2RGB, imwrite
from PIL.ImageGrab import grab
import numpy as np
import pygame

####################################################
# Record your races in Trackmania : 
# Get screen while you drive, and record your inputs.
# You have 3 seconds to switch in game when you start this script.
# To end your record, press "b" in the controler 
####################################################

class Record:

    def __init__(self, dataset_folder):
        # Create folders, files and save paths
        self.path = './dataset/' + dataset_folder
        if not isdir(self.path):
            mkdir(self.path)

        self.fileDataRaw = self.path + '/dataRaw.json'
        if not exists(self.fileDataRaw):
            with open(self.fileDataRaw, 'w') as outfile:
                dump([], outfile)

        self.fileConfig = self.path + '/config.json'
        if not exists(self.fileConfig):
            with open(self.fileConfig, 'w') as outfile:
                dump({"num": 0}, outfile)

        self.pathImages = self.path + '/images/'
        if not isdir(self.pathImages):
            mkdir(self.pathImages)
        
        self.data = []
        self.run = True
    
    def record(self):
        self.ControllerListener()

        print('3 secondes before starting record.')
        sleep(3)
        print('Start record !')

        # Record run
        while self.run:
            start = time()

            # Record inputs and sreens
            inputs = self.getInputController()
            image = self.getScreen()

            infos = {"image": image, "inputs" : inputs}
            self.data.append(infos)

            # print speed
            sleep(0.01)
            print('Recording : ', round(1. / (time() - start)), " Fps.", end="\r")

        print('End record.')
    
    def saveRecord(self):
        # Load old dataset to complete.
        with open(self.fileDataRaw) as json_file:
            dataRaw = load(json_file)

        with open(self.fileConfig) as json_file:
            config = load(json_file)

        # Print infos
        nbImgToSave = len(self.data)
        print('Saving ' + str(nbImgToSave) + ' images.')

        # Save infos
        i=config["num"]
        for img_number, infos in enumerate(self.data):
            image_name = self.pathImages + 'img' + str(i) + '.png'
            imwrite(image_name,  infos["image"])
            dataRaw.append({"imgFilename": image_name, "target" : infos["inputs"]})
            i += 1

            print('     ' + str(round((img_number/nbImgToSave)*100)) + '%', end="\r")
            sleep(0.01)

        print('Save images done.')

        print('Saving data raw and config...')
        with open(self.fileDataRaw, 'w') as outfile:
            dump(dataRaw, outfile)

        with open(self.fileConfig, 'w') as outfile:
            dump({"num": i}, outfile)

        print('Save data raw and config done')
        print('Saving Record finished !')

    def getInputController(self):
        pygame.event.pump()

        # get the inputs. 
        # On a xbox controller, axis(0) is left joystick, button(0) is "A" button and button(2) is "X" button.
        direction = self.controller.get_axis(0)
        accel = self.controller.get_button(0)
        brake = self.controller.get_button(2)

        # If we press "B" button on a xbox controller, we stop recording.
        self.run = not self.controller.get_button(1)
        if not self.run :
            pygame.quit()

        return  [round(direction, 2), accel, brake]

    def getScreen(self):
        screen = grab()
        image = cvtColor(np.array(screen), COLOR_BGR2RGB)

        # Resize to 50 percent of original dimensions
        width = int(image.shape[1] * 50 / 100)
        height = int(image.shape[0] * 50 / 100)

        imageScale = resize(image, (width, height))

        return imageScale
    
    def ControllerListener(self):
        pygame.init()
        self.controller = None

        for i in range(0, pygame.joystick.get_count()):
            if pygame.joystick.Joystick(i).get_name().find("Xbox") != -1:
                self.controller = pygame.joystick.Joystick(i)
        if self.controller is None:
            exit("Xbox controller not find.")
        
        self.controller.init()