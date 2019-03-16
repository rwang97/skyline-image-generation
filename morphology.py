import cv2 as cv
import numpy as np
import os

def zero_out(img):
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

    img = cv.normalize(img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # Otsu's thresholding
    ret, thresh3 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # adaptive mean
    thresh4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
    thresh5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)
    cv.imwrite("thresh1.jpg", thresh1)
    cv.imwrite("thresh2.jpg", thresh2)
    cv.imwrite("thresh3.jpg", thresh3)
    cv.imwrite("thresh4.jpg", thresh4)
    cv.imwrite("thresh5.jpg", thresh5)
    return

def detect_edges(img):
    #img = cv2.imread(imageSource)
    #img = cv2.cvtColor(imageSource,cv2.COLOR_BGR2GRAY).astype(float)

    edge_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    edge_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    edge = np.sqrt(edge_x**2 + edge_y**2)

    cv.imwrite("edge.jpg", edge)
    return edge

def rgb_to_gray(imageSource):
    img = cv.imread(imageSource)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float)
    cv.imwrite("gray.jpg", img)
    return img

def mor_closing():
    # change directory to data
    os.chdir('test_img')
    imageSource = 'daytime_skyline.jpg'

    img = rgb_to_gray(imageSource)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imwrite("closing.jpg", closing)

    img = detect_edges(closing)
    zero_out(img)
    return


if __name__ == '__main__':
    # os.chdir('test_img')
    # imageSource = 'daytime_skyline.jpg'
    #
    # img = rgb_to_gray(imageSource)
    # edge = detect_edges(img)
    # zero_out(edge)

    mor_closing()
    pass