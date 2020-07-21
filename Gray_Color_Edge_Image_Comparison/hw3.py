#import all the required python libraries 
import cv2
import numpy as np
import os
from os import listdir
from os.path import join, isfile
from matplotlib import pyplot as plt
import math

path = 'ST2MainHall4/'
gmax = 255
nimgs = 0
bin = 36
postfixgray = 4000;
postfixcolor = 4000;
ghistograms=[]
rgbhistograms=[]

def HistogramIntersection(H1, H2):
    sum_dividend = 0.0
    sum_divisor  = 0.0
    for i in range(0, bin):
        sum_dividend = sum_dividend + min(H1[i], H2[i])
        sum_divisor  = sum_divisor  + max(H1[i], H2[i])
    return sum_dividend/sum_divisor

def ChiSquaredMeasure(H1, H2):
    summation = 0.0
    for i in range(0, bin):
        #summation constraint i:H1+H2 > 0
        if (H1[i] + H2[i] > 0):
            summation += (math.pow(H1[i] - H2[i], 2) / (H1[i] + H2[i]))
    return summation

def normalize(ChiSquare):
    minimum = np.amin(ChiSquare)
    maximum = np.amax(ChiSquare)
    ChiSquare = (gmax - gmax * (ChiSquare - minimum) / (maximum - minimum))
    return ChiSquare

def BuildGrayHistograms(imgfile):
    global ghistograms
    global postfixgray
    postfixgray += 1;

    imgfilename = join(path, imgfile)
    img = cv2.imread(imgfilename, 0)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

    mask = np.uint8(cv2.Canny(img, 100, 200)) 
    (_, angle) = cv2.cartToPolar(sobelx, sobely, angleInDegrees = True)
    indices = np.uint8(np.round(np.divide(angle, 10)))

    histogram = cv2.calcHist([indices], [0], mask, [bin], [0,bin])
    ghistograms = [*ghistograms, histogram]

    PlotAndSaveHistogram(histogram, postfixgray, 0)

def BuildColorHistograms(imgfile):
    global rgbhistograms
    global postfixcolor
    postfixcolor += 1;

    imgfilename = join(path, imgfile)
    img = cv2.imread(imgfilename)

    b, g, r = cv2.split(cv2.GaussianBlur(img, (9, 9), 0))
    sobelrx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize = 5)
    sobelry = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize = 5)
    sobelgx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize = 5)
    sobelgy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize = 5)
    sobelbx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize = 5)
    sobelby = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize = 5)

    mask = np.uint8(cv2.Canny(r, 100, 200) + cv2.Canny(g, 100, 200) + cv2.Canny(b, 100, 200))
    (_, angle) = cv2.cartToPolar((sobelrx + sobelgx + sobelbx), (sobelry + sobelgy + sobelby), angleInDegrees = True)
    indices = np.uint8(np.round(np.divide(angle, 10)))
    
    histogram = cv2.calcHist([indices], [0], mask, [bin], [0,bin])
    rgbhistograms = [*rgbhistograms, histogram]

    PlotAndSaveHistogram(histogram, postfixcolor, 1)

def Quiver():
    #reference: https://matplotlib.org/examples/pylab_examples/quiver_demo.html
    print("Showing Quiver Plot")
    imgfile = "ST2MainHall4/ST2MainHall4001.jpg"
    img = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img = cv2.GaussianBlur(img,(7,7),0)

    X,Y = np.meshgrid(np.arange(w), np.arange(h)) #size of img
    sobelx = np.uint8(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    sobely = np.uint8(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))
    scale = 50
    M = np.hypot(sobelx, sobely)
    
    plt.figure()
    plt.title('Gradient Visualization of ST2MainHall4001.jpg')
    #Q = plt.quiver(X[::scale,::scale], Y[::scale,::scale], sobelx[::scale,::scale], sobely[::scale,::scale], pivot='mid', units='width')
    Q = plt.quiver(X[::scale,::scale], Y[::scale,::scale], sobelx[::scale,::scale], sobely[::scale,::scale], M, width = 1, units = 'dots', pivot = 'mid')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, '', labelpos='E', coordinates='figure')
    plt.scatter(X[::scale, ::scale], Y[::scale, ::scale], color = 'k', s = 1)
    plt.show()

def PlotAndSaveHistogram(histogram, postfix, grayorcolor):
    plt.plot(histogram)
    if grayorcolor == 0:
        plt.savefig('img/gray/grayhistogram' + str(postfix) + '.png')
    else:
        plt.savefig('img/color/colorhistogram' + str(postfix) + '.png')
    plt.clf()
    
def DisplayAndSave(img, title, name):
    print("Saving and Displaying: " + title)
    plt.imshow(img, cmap = 'hot', interpolation = 'nearest')
    plt.title(title)
    plt.colorbar()
    plt.savefig('img/' + name)    
    plt.show()

def main():
    Quiver() #display gradient visualization, close window to continue

    #read images
    imagefiles = [imgfile for imgfile in listdir(path) if isfile(join(path, imgfile))]
    imagefiles.sort()
    nimgs = len(imagefiles)

    print('Plotting and saving gray and color histograms')
    for imgfile in imagefiles:
        BuildGrayHistograms(imgfile)
        BuildColorHistograms(imgfile)

    #histogram comparison arrays
    GrayIntersection  = np.zeros((nimgs, nimgs))
    GrayChiSquare     = np.zeros((nimgs, nimgs))
    ColorIntersection = np.zeros((nimgs, nimgs))
    ColorChiSquare    = np.zeros((nimgs, nimgs))

    for i in range(0, nimgs - 1):
        #correct diagonal
        GrayIntersection[i,i]  = 1.0
        ColorIntersection[i,i] = 1.0
        GrayIntersection[nimgs - 1, nimgs - 1]   = 1.0
        ColorIntersection[nimgs - 1, nimgs - 1]  = 1.0
        for j in range(i + 1, nimgs):
            GrayIntersection[i,j]  = HistogramIntersection(ghistograms[i],ghistograms[j])
            GrayIntersection[j,i]  = GrayIntersection[i,j]
            ColorIntersection[i,j] = HistogramIntersection(rgbhistograms[i],rgbhistograms[j])
            ColorIntersection[j,i] = ColorIntersection[i,j]
            GrayChiSquare[i,j]     = ChiSquaredMeasure(ghistograms[i],ghistograms[j])
            GrayChiSquare[j,i]     = GrayChiSquare[i,j]
            ColorChiSquare[i,j]    = ChiSquaredMeasure(rgbhistograms[i],rgbhistograms[j])
            ColorChiSquare[j,i]    = ColorChiSquare[i,j]
    
    #normalize imgs
    GrayIntersection  = gmax * GrayIntersection 
    ColorIntersection = gmax * ColorIntersection 
    GrayChiSquare     = normalize(GrayChiSquare)
    ColorChiSquare    = normalize(ColorChiSquare)

    DisplayAndSave(GrayIntersection, 'Gray Intersection Comparison', 'GrayIntersection.png')
    DisplayAndSave(GrayChiSquare, 'Gray Chi Square Comparison', 'GrayChiSquare.png')
    DisplayAndSave(ColorIntersection, 'Color Intersection Comparison', 'ColorIntersection.png')
    DisplayAndSave(ColorChiSquare, 'Color Chi Square Comparison', 'ColorChiSquare.png')

if __name__ == '__main__':
   main()