# Alec Webb CS682 Homework2-problem1

#Requires: image requested for evaluation be a COLOR image as specified by 
#		   part 1 of problem 1.1
#Effects: produce a histogram evaluation of the images color channels
#		  as requested by problem 1.1 part 2. Upon closing this window will then
#		  produce a picture in which a gray rectangle hovers over an 11x11 region
#		  of pixels and provides data output surrounding that box as outlined in
#		  problem 1.1 parts 3a,3b,3c,3d, this picture is the one identified
#		  by the user in the initial prompt

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import *

class takeInput(object):
    def __init__(self, requestMessage):
        self.root = Tk()
        self.string = ''
        self.frame = Frame(self.root)
        self.frame.pack()        
        self.acceptInput(requestMessage)

    def acceptInput(self, requestMessage):
        r = self.frame
        k = Label(r, text = requestMessage)
        k.pack(side = 'left')
        self.e = Entry(r, text = 'Name')
        self.e.pack(side = 'left')
        self.e.focus_set()
        b = Button(r,text='okay',command=self.gettext)
        b.pack(side='right')

    def gettext(self):
        self.string = self.e.get()
        self.root.destroy()

    def getString(self):
        return self.string

    def waitForInput(self):
        self.root.mainloop()

def getText(requestMessage):
    msgBox = takeInput(requestMessage)
    #loop until the user makes a decision and the window is destroyed
    msgBox.waitForInput()
    return msgBox.getString()

#globals
img = None
image = None

#draws square curve border
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
 
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

#computes and displays 3 channel histogram
def ColorChannelHistogram(image):
    img = cv2.imread(image)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img], [i] , None, [256], [0, 256])
        plt.plot(histr,color = col)
        plt.xlim([0, 256])
    plt.show()

# mouse callback function
def interactive_window(event,x,y,flags,param):
    global ix,iy,drawing, mode, img, image

    if event==cv2.EVENT_MOUSEMOVE:
        ix, iy = x, y
        img = cv2.imread(image)
        draw_border(img, (ix - 7, iy - 7),(x + 7, y + 7), (255, 255, 255), 1, 1, 10)
        print('---------------')

        #11x11 rectangle window, offset is 5 pixels +/- p-coords
        #x1,y1-----------|
        #|				 |
        #|				 |
        #|		 p  	 |
        #|				 |
        #|				 |
        #--------------x2,y2
        cv2.rectangle(img, (ix - 5, iy - 5), (x+5, y + 5), (128,128,128), 1)

        #COORDS FOR P
        font = cv2.FONT_HERSHEY_PLAIN
        strXY = 'x:' + str(x) + ',y:' + str(y)
        print(strXY)
        cv2.putText(img, strXY, (ix - 50, iy - 30), font, 0.75, (0, 255, 0), 1)

        #RGB VALUE EXTRACTION
        b, g, r = img[y, x]
        strRGB = 'R:' + str(r) + ',G:' + str(g) + ',B:' + str(b)
        print(strRGB)
        cv2.putText(img, strRGB, (ix - 50, iy - 15), font, 0.75, (0, 255, 0), 1)

        #INTENSITY CALCULATION
        intensity = (r + g + b)/3
        strIntensity = 'Intensity:' + str(intensity)
        print(strIntensity)
        cv2.putText(img, strIntensity, (ix - 50, iy + 20), font, 0.75, (0, 255, 0), 1)

        #MEAN/STDDEV EXTRACT 11x11 pixels from region of interest
        region = img[y - 5:y + 6, x - 5:x + 6]
        mean = np.mean(region, axis=(0, 1))
        std = np.std(region, axis=(0, 1))
        print('Mean[B:G:R]:' + str(mean))
        print('STDDEV[B:G:R]:'+ str(std))
        cv2.putText(img, 'Mean[B:G:R]:' + str(mean), (ix - 50, iy + 30), font, 0.75, (0, 255, 0), 1)
        cv2.putText(img, 'STDDEV[B:G:R]:'+ str(std), (ix - 50, iy + 40), font, 0.75, (0, 255, 0), 1)
        ix = x
        iy = y
    return x, y

#runs a loop of the image for the window to be placed upon
def RunImage(image):
    global img
    img = cv2.imread(image)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image', interactive_window)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # wait for ESC key to exit
            break

    cv2.destroyAllWindows()

def main():
    global image
    #allow image selection by way of prompt
    image = getText('enter the file name') #'sumer.png'
    ColorChannelHistogram(image)
    RunImage(image)

if __name__ == '__main__':
    main()