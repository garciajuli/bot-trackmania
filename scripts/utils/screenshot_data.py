import math
import time
import threading
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

####################################################
# Class for the data screen shoted : 
# We compute the wall distance and the speed.
####################################################

class ScreenData():
    def __init__(self):
        self.model =  tf.keras.models.load_model('./model/digits/vitesse_model3.h5')
        self.maskCar = cv2.imread('maskCar.png', 0)

        self.distances = []
        self.pos = []
        self.speed = 0

        self.speedDigits = [0]*3
        self.digits = [0]*3
        self.isReadSpeed = False

        # vision data init
        self.coefA = [10, 5, 2, 1.5, 1, 0.8, 0.6]
        self.signes = [-1, 1]
        self.xLine = 480
        self.affineFunc = lambda a, x, b : round(a * x + b)
        self.distanceTmp = [0] * (len(self.coefA)*len(self.signes))

        self.mask = None

    def getInfos(self):
        speedThread = threading.Thread(target=self.getSpeed)
        distancesThread = threading.Thread(target=self.getWallDistance)

        speedThread.start()
        distancesThread.start()

        speedThread.join()
        distancesThread.join()

        return  int(self.speed), self.distances, self.pos

    def setImage(self, image):
        self.orig = image

    def getWallDistance(self):
        ts = time.time()
        image = self.orig.copy()

        #only for simple track
        l_h, l_s, l_v, u_h, u_s, u_v = 10, 0, 170, 255, 80, 255

        lo = np.array([l_h, l_s, l_v])
        hi = np.array([u_h, u_s, u_v])

        imageHSV=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        imageBlur=cv2.blur(imageHSV, (8, 8))
        mask=cv2.inRange(imageBlur, lo, hi)
        # mask=cv2.erode(mask, None, iterations=2)
        # mask=cv2.dilate(mask, None, iterations=2)

        mask = cv2.bitwise_not(mask)
        mask=cv2.bitwise_and(mask, mask, mask=self.maskCar)
        self.mask = cv2.bitwise_not(mask)

        elements=cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        distances= []
        if elements :
            c=max(elements, key=cv2.contourArea)
            distances, pos = self.findDistanceWall(c)

        self.distances = distances
        self.pos = pos
        # print("wall : ", time.time() - ts)
        return distances, pos
                  
    def getPredict(self, digit, index):
        self.speedDigits[index] = np.argmax(self.model.predict(self.setupData(digit))[0])

    def getSpeed(self):
        image = self.orig.copy()
        speed = image[430:495, 400:550]

        digits = self.getDigits(speed)

        digitsThread = []
        self.isReadSpeed = True

        for i, digit in enumerate(digits):
            thread = threading.Thread(target=self.getPredict, args=(digit, i,))
            digitsThread.append(thread)
            thread.start()
            
        for thread in digitsThread:
            thread.join()
        
        self.speed = self.speedDigits[0] * 100 + self.speedDigits[1] * 10 + self.speedDigits[2]
    
    def stopReadSpeed(self):
        self.isReadSpeed = False
        self.threadSpeed.join()

    def updateSpeed(self):
        self.speed = self.speedDigits[0] * 100 + self.speedDigits[1] * 10 + self.speedDigits[2]
        

    def findDistanceWall(self, contour):
        contourBlack = np.zeros((self.orig.shape[0], self.orig.shape[1]))

        cv2.drawContours(contourBlack, [contour], -1, (255, 255, 255), 2)

        colors = []
        i = 0
        lines = np.zeros((self.orig.shape[0], self.orig.shape[1]))
        for signe in self.signes:
            for a in self.coefA:
                x = self.xLine*signe
                y = self.affineFunc(a, x, 0)*signe            
                lines = cv2.line(lines , (x+480, 540-y), (480, 530), 255-i, 1)
                colors.append(255-i)
                i += 1

        intersection =  cv2.bitwise_and(contourBlack, lines)

        result = []
        for color in colors:
            result.append([])

        positions = np.where(intersection >= 255-i)
        positions = zip(positions[1], positions[0])

        for pos in positions:
            c = int(intersection[pos[1], pos[0]])
            result[abs(c-255)].append(pos)

        positionRes = []   
        for i, line in enumerate(result):
            if len(line) > 0:
                if len(line) > 1:
                    line = sorted(line, key=lambda x: round(math.sqrt((480 - x[0])**2 + (540 - x[1])**2)), reverse=True)
                positionRes.append(line[0])
        
        positionRes = sorted(positionRes, key=lambda x: x[0])

        distanceRes = []
        for pos in positionRes:
            distanceRes.append(round(math.sqrt((480 - pos[0])**2 + (540 - pos[1])**2)))

        return distanceRes, positionRes
    
    def intersectionWithFunc(self, img1, img2, addX, index):
        intersection =  cv2.bitwise_and(img1, img2)
        result = np.where(intersection == 255)
        
        intersX = result[1] + addX
        intersY = result[0]

        inters = sorted(zip(intersX, intersY), key=lambda x: round(math.sqrt((480 - x[0])**2 + (540 - x[1])**2)), reverse=True)        

        self.distanceTmp[index] = inters[0]


    def getDigits(self, img):
        digits = []

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        areaArray = []
        for _, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)
        
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        
        xArray = []
        indexC = 0
        indexMax = 3
        while indexC < indexMax:
            drawble = True
            c = sorteddata[indexC][1]
            leftmost, _ = tuple(c[c[:,:,0].argmin()][0])
            for xC in xArray:
                if xC - 10 < leftmost < xC + 10:
                    indexMax += 1
                    drawble = False
                    break

            if drawble:
                digits.append(self.boxedDigit(img, c, (self.orig, 430, 400)))
                xArray.append(leftmost)
            indexC += 1
        
        sorteddata = sorted(zip(xArray, digits), key=lambda x: x[0])
        resDigits = []
        for d in sorteddata:
            resDigits.append(d[1])

        return resDigits

    def boxedDigit(self, img, c, infoOrig):
        leftmost, _ = tuple(c[c[:,:,0].argmin()][0])
        rightmost, _ = tuple(c[c[:,:,0].argmax()][0])
        _, topmost = tuple(c[c[:,:,1].argmin()][0])
        _, bottommost = tuple(c[c[:,:,1].argmax()][0])

        imgOrig, y, x = infoOrig
        digit = imgOrig[(topmost+y)-5:(bottommost+y)+5, (leftmost+x)-5:(rightmost+x)+5]

        return digit
    
    def setupData(self, img):
        digit = cv2.resize(img, (45, 32))
        img_array = tf.expand_dims(digit, 0)
        inputs_dataset = tf.data.Dataset.from_tensor_slices(img_array).batch(1)
        return inputs_dataset