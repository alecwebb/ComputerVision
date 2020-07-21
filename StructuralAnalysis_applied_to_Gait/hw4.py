import cv2
import math
import csv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import join, isfile

gmax = 255
filename = ""
distanceTransforms = []
allcontours = []
allfeatures = [] 
originalarea = []
originalperimiter = [] 
convexHullArea = []
convexHullPerimeter = []
alldeficitsarea = []
alldeficits = []
firstOrderMoments = []
convexFirstOrderMoments = []
secondOrderMoments = []
convexSecondOrderMoments = []

def distanceTransformEuclidean(img):
    disttransform = cv2.distanceTransform(img, distanceType = cv2.DIST_L2, maskSize = 5)
    disttransform = disttransform / disttransform.max()
    cv2.imwrite('results/EuclideanDistTransform/' + filename, gmax * disttransform)
    return disttransform

def chamferMatching(contours, distancetransforms):
    distancetransforms[np.where(distancetransforms == gmax)] = 1
    distancemeasure = np.sum(distancetransforms[contours[:,0,1], contours[:,0,0]])
    return distancemeasure

def calcContour(grayimg):
    _, contours, hierarchy = cv2.findContours(grayimg, 1, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite('results/Contours/' + filename, cv2.drawContours(originalimg.copy(), contours, -1, (0, gmax, 0), 1))
    return contours

def polygonalApproximation(contour):
    cv2.imwrite('results/PolyApprox/' + filename, cv2.drawContours(originalimg.copy(), [(cv2.approxPolyDP(contour, (cv2.arcLength(contour, True) * 0.01), True))],-1, (gmax, 0, gmax), 1))

def distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def findArea(x, y, z):
    dist = (distance(x, y) + distance(y, z) + distance(z, x)) / 2
    return math.sqrt(dist * (dist - distance(x, y)) * (dist - distance(y, z)) * (dist - distance(z, x)))

def convexHulll(contour):
    img = originalimg.copy()
    defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints = False))
    areaofdefects=0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0]) 
        areaofdefects += findArea(start, end, far)
        cv2.line(img, start, end, [0, gmax, 0], 1)
        cv2.circle(img, far, 1, [0, 0, gmax], -1)
        
    cv2.imwrite('results/Convexity/' + filename, img)
    return defects.shape[0], areaofdefects

def calculateAPM(contour):
    return (cv2.contourArea(contour), cv2.arcLength(contour, True), cv2.moments(contour))

def curvature(contour):
    angle = 2 * (
        (4 * np.append(contour[4:,0,0], contour[0:4, 0, 0]) - 3 * contour[:, 0, 0] - np.append(contour[len(contour) - 4:, 0, 0], contour[0:len(contour) - 4, 0, 0]) )/2 * 
        (contour[:, 0, 1] + np.append(contour[len(contour) - 4:, 0, 1], contour[0:len(contour) - 4, 0, 1]) - 2 * np.append(contour[4:, 0, 1], contour[0:4, 0, 1]) )/2 - 
        (4 * np.append(contour[4:, 0, 1], contour[0:4, 0, 1]) - 3 * contour[:, 0, 1] - np.append(contour[len(contour) - 4:, 0, 1], contour[0:len(contour) - 4, 0, 1]) )/2 * 
        (contour[:, 0, 0] + np.append(contour[len(contour) - 4:, 0, 0], contour[0:len(contour) - 4, 0, 0]) - 2 * np.append(contour[4:,0,0], contour[0:4, 0, 0]) )/2 
        ) / ( np.power((
            (4 * np.append(contour[4:, 0, 0], contour[0:4, 0, 0]) - 3 * contour[:, 0, 0] - np.append(contour[len(contour) - 4:, 0, 0], contour[0:len(contour) - 4, 0, 0]) )/2 * 
            (4 * np.append(contour[4:, 0, 0], contour[0:4, 0, 0]) - 3 * contour[:, 0, 0] - np.append(contour[len(contour) - 4:, 0, 0], contour[0:len(contour) - 4, 0, 0]) )/2 + 
            (4 * np.append(contour[4:, 0, 1], contour[0:4, 0, 1]) - 3 * contour[:, 0, 1] - np.append(contour[len(contour) - 4:, 0, 1], contour[0:len(contour) - 4, 0, 1]) )/2 * 
            (4 * np.append(contour[4:, 0, 1], contour[0:4, 0, 1]) - 3 * contour[:, 0, 1] - np.append(contour[len(contour) - 4:, 0, 1], contour[0:len(contour) - 4, 0, 1]) )/2), 1.5) )

    curvature = np.array( gmax - (np.absolute(angle)) * 255.0 / np.amax(angle), dtype = np.uint8 )
    img = grayimg.copy()
    img[contour[:, 0, 1], contour[:, 0, 0]] = curvature
    cv2.imwrite('results/Curvature/' + filename, cv2.applyColorMap(img, cv2.COLORMAP_HOT))
    return curvature 

def calcLocalMaxima(contour, curvature):
    localmaxima = np.where(
        (curvature > np.append(curvature[1:], curvature[0: 1])) & 
        (curvature > np.append(curvature[len(contour) - 1:], curvature[0:len(contour) - 1])) & 
        (curvature > np.append(curvature[2:], curvature[0:2])) & 
        (curvature > np.append(curvature[len(contour) - 2:], curvature[0:len(contour) - 2])) & 
        (curvature > np.append(curvature[2:], curvature[0:2])) & 
        (curvature > np.append(curvature[len(contour) - 2:], curvature[0:len(contour) - 2]))
        )
    localmaximaimg = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
    for i in localmaxima[0]:
        localmaximaimg = cv2.circle(localmaximaimg, (contour[:, 0, 0][i], contour[:, 0, 1][i]), 1, (gmax, 0, 0), -1)

    cv2.imwrite('results/LocalMaxima/' + filename, localmaximaimg)

def displayChamfer(distanceTransforms, allcontours):
    chamfer = np.zeros((len(distanceTransforms), len(distanceTransforms)), dtype = np.float)
    for i in range(0, len(distanceTransforms)):
        for j in range(0, len(distanceTransforms)):
            chamfer[i][j] = chamferMatching(allcontours[i], distanceTransforms[j])
            chamfer[j][i] = chamfer[i][j]

    plt.imshow(np.array(chamfer * 255.0 / chamfer.max(), dtype = np.uint8), interpolation='none') 
    plt.set_cmap('bone')
    plt.suptitle('Chamfer Matching', fontsize = 10)
    plt.savefig('results/ChamferMatchingGraph.png')
    plt.show()
    plt.gcf().clear()

def displayImages():
    print('Displaying calculations for image: 00000048.png')
    displayImage('GaitImages/', '00000048.png', 'GaitImage')
    displayImage('results/Contours/', '00000048.png', 'Contour')
    displayImage('results/PolyApprox/', '00000048.png', 'Polygonal Approximation')
    displayImage('results/Convexity/', '00000048.png' , 'Convexity and Deficits')
    displayImage('results/Curvature/', '00000048.png', 'Curvature: Higher = Hotter Red')
    displayImage('results/LocalMaxima/', '00000048.png', 'Local Maxima')
    displayImage('results/EuclideanDistTransform/', '00000048.png', 'Euclidian Distance Transform')
    print('Displaying calculations for image: 00000049.png')
    displayImage('GaitImages/', '00000049.png', 'GaitImage')
    displayImage('results/Contours/', '00000049.png', 'Contour')
    displayImage('results/PolyApprox/', '00000049.png', 'Polygonal Approximation')
    displayImage('results/Convexity/', '00000049.png' , 'Convexity and Deficits')
    displayImage('results/Curvature/', '00000049.png', 'Curvature: Higher = Hotter Red')
    displayImage('results/LocalMaxima/', '00000049.png', 'Local Maxima')
    displayImage('results/EuclideanDistTransform/', '00000049.png', 'Euclidian Distance Transform')

def displayImage(path, fname, title):
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 88 * 3, 128 * 3)
    cv2.imshow(title, cv2.imread(path + fname))
    cv2.waitKey()
    cv2.destroyAllWindows()

imagefiles = [imgfile for imgfile in listdir('GaitImages/') if isfile(join('GaitImages/', imgfile))]
imagefiles.sort()

for imgfile in imagefiles:
    imgfilename = join('GaitImages/', imgfile)
    filename = imgfile    
    img = cv2.imread(imgfilename)
    grayimg = cv2.imread(imgfilename , 0)
    originalimg = img.copy()

    print('Processing:' + imgfilename)
    
    _, threshold = cv2.threshold(grayimg, 127, gmax, cv2.THRESH_BINARY)
    contours = calcContour(threshold)
    contour = contours[len(contours)-1]
    allcontours.append(contour)
    polygonalApproximation(contour)
    calcLocalMaxima(contour, curvature(contour))
    distanceTransforms.append(distanceTransformEuclidean(threshold))
    numberofdeficits, areaofdeficits = convexHulll(contour)
    alldeficits.append(numberofdeficits)
    alldeficitsarea.append(areaofdeficits)
    area, perimeter, Moments = calculateAPM(contour)
    originalarea.append(area)
    originalperimiter.append(perimeter)
    firstOrderMoments.append([Moments['m10'],Moments['m01']])
    secondOrderMoments.append([Moments['m20'],Moments['m02'],Moments['m11']])
    area, perimeter, Moments = calculateAPM(cv2.convexHull(contour))
    convexHullArea.append(area)
    convexHullPerimeter.append(perimeter)
    convexFirstOrderMoments.append([Moments['m10'], Moments['m01']])
    convexSecondOrderMoments.append([Moments['m20'], Moments['m02'], Moments['m11']])
    computedfeatures                            = {}
    computedfeatures['filename']                = filename
    computedfeatures['Original Area']           = area
    computedfeatures['Original Perimeter']      = perimeter
    computedfeatures['Original m10']            = Moments['m10']
    computedfeatures['Original m01']            = Moments['m01']
    computedfeatures['Original m11']            = Moments['m11']
    computedfeatures['Original m20']            = Moments['m20']
    computedfeatures['Original m02']            = Moments['m02']
    computedfeatures['Convex Hull Area']        = area
    computedfeatures['Convex Hull Perimeter']   = perimeter
    computedfeatures['Convex Hull m10']         = Moments['m10']
    computedfeatures['Convex Hull m01']         = Moments['m01']
    computedfeatures['Convex Hull m11']         = Moments['m11']
    computedfeatures['Convex Hull m20']         = Moments['m20']
    computedfeatures['Convex Hull m02']         = Moments['m02']
    computedfeatures['# of Deficits']           = numberofdeficits
    computedfeatures['Area of Deficits']        = areaofdeficits
    allfeatures.append(computedfeatures)

with open('results/computedfeatures.csv', 'w') as output:
    dict_writer = csv.DictWriter(output, allfeatures[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(allfeatures)

displayImages()
displayChamfer(distanceTransforms, allcontours)