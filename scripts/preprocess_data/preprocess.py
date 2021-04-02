from ..utils.screenshot_data import ScreenData
from os.path import exists, isdir
from json import dump, load
from cv2 import imread
from time import sleep
import numpy as np

####################################################
# Pre process the dataset : 
# Compute wall distances and speed to create a new database with these informations.
####################################################

class Dataset:

    def __init__(self, dataset_folder):
        # Load and create dataset files
        self.path = './dataset/' + dataset_folder
        if not isdir(self.path):
            exit("Data not found for this name and device.")

        self.fileDataset = self.path + '/dataset.json'
        if not exists(self.fileDataset):
            with open(self.fileDataset, 'w') as outfile:
                dump([], outfile)

        self.fileDataRaw = self.path + '/dataRaw.json'
        if exists(self.fileDataRaw):
            with open(self.fileDataRaw) as json_file:
                self.dataRaw = load(json_file)
        else:
            exit("Data raw not found for this name and device.")

        self.pathImages = self.path + '/images/'
        if not isdir(self.pathImages):
            exit("Folder images not found for this name and device.")
        
        self.dataImage = ScreenData()
    
    def processingData(self):
        with open(self.fileDataset) as json_file:
            self.dataset = load(json_file)

        data_number = len(self.dataRaw)

        # Pre process the data
        print('Preprocess sceenshots...')
        for enum_data, data in enumerate(self.dataRaw):
            imagePath = data["imgFilename"]
            image = imread(imagePath)

            self.dataImage.setImage(image)
            # Compute wall distance and speed
            speed, distances, _ = self.dataImage.getInfos()

            # Flip data and target to double the size of our database.
            distancesFlip = np.flip(distances, 0).tolist()
            targetFlip = [data["target"][0]*(-1), data["target"][1], data["target"][2]]

            # append infos to the dataset JSON
            self.dataset.append({"wallDistances": distances, "speed": speed, "target": data["target"]})
            self.dataset.append({"wallDistances": distancesFlip, "speed": speed, "target": targetFlip})

            # Print  advancement
            print('     ' + str(round((enum_data/data_number)*100)) + '%', end="\r")
            sleep(0.01)

    def saveDataset(self):
        print('Write json Dataset...')
        with open(self.fileDataset, 'w') as outfile:
            dump(self.dataset, outfile)
        print('Done.')
