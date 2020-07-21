# Alec Webb CS682 Homework2-problem2
from os import listdir
from os.path import join, isfile
from itertools import product

import matplotlib.pyplot as pyplot
import numpy as np
import math
import cv2

path = 'ST2MainHall4/'
histograms = []
gmax = 255
nimgs = 0
bin = 512

def BuildHistograms():
    global nimgs
    global histograms

    print("Loading image files...")
    imagefiles = [imgfile for imgfile in listdir(path) if isfile(join(path, imgfile))]
    imagefiles.sort()
    nimgs = len(imagefiles)

    for imgfile in imagefiles:
        imgfilename = join(path, imgfile)
        img = cv2.imread(imgfilename)
        b, g, r = cv2.split(img)
        #x << k == x multiplied by 2 to the power of k
        #x >> k == x divided by 2 to the power of k
        colortoindex = ((b >> 5) << 6) + ((g >> 5) << 3) + (r >> 5)
        (rows, cols) = colortoindex.shape
        indices = colortoindex.reshape([rows * cols, 1])
        (histogram, _) = np.histogram(indices, bins = range(0, bin + 1))
        histograms = [*histograms, histogram]

def ComputeDisplayIntersection():
    print("Computing histogram intersection...")
    ComputedIntersection = np.ones((nimgs, nimgs))
    for i in range(0, nimgs - 1):
        for j in range(i + 1, nimgs):
            ComputedIntersection[i, j] = HistogramIntersection(histograms[i], histograms[j])
            ComputedIntersection[j, i] = ComputedIntersection[i, j]

    #convert to range 0 to gmax
    IntersectionImage = (gmax * ComputedIntersection)
    Display(IntersectionImage, 'Histogram Intersection')

def ComputeDisplayChiSquare():
    print("Computing chi square measure...")
    ComputedChiSquare = np.zeros((nimgs, nimgs))
    for i in range(0, nimgs - 1):
        for j in range(i + 1, nimgs):
            ComputedChiSquare[i, j] = ChiSquaredMeasure(histograms[i], histograms[j])
            ComputedChiSquare[j, i] = ComputedChiSquare[i, j]

    #convert to range 0 to gmax
    ChiSqureImage = (gmax - (gmax * (ComputedChiSquare)) / (np.amax(ComputedChiSquare)))
    Display(ChiSqureImage, 'Chi Squared Measure')

def HistogramIntersection(H1, H2):
    sum_dividend = 0.0
    sum_divisor = 0.0
    for i in range(0, bin):
        sum_dividend = sum_dividend + min(H1[i], H2[i])
        sum_divisor = sum_divisor + max(H1[i], H2[i])
    return sum_dividend/sum_divisor

def ChiSquaredMeasure(H1, H2):
    summation = 0.0
    for i in range(0, bin):
        #summation constraint i:H1+H2 > 0
        if (H1[i] + H2[i] > 0):
            summation += (math.pow(H1[i] - H2[i], 2) / (H1[i] + H2[i]))
    return summation

def Display(img, title):
    pyplot.imshow(img)
    pyplot.gray()
    pyplot.suptitle(title)
    pyplot.colorbar()    
    pyplot.show()

def main():
    BuildHistograms()
    ComputeDisplayIntersection()
    ComputeDisplayChiSquare()
    print("Execution complete")

if __name__ == '__main__':
   main()
