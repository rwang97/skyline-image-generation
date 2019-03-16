import cv2
import numpy as np
import os

# https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
def detect_edges():
    os.chdir('data')
    imageSource = 'pic_name.jpg'
    img = cv2.imread(imageSource)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)

    edge_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    edge_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    edge = np.sqrt(edge_x**2 + edge_y**2)

    cv2.imwrite("out_name.jpg", edge)

def rename_img():
    # change directory to data
    os.chdir('data')
    # rename all the images to indices
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            os.rename(image, str(i) + ".jpg")
            i += 1

def flip_img():
    # change directory to data
    os.chdir('data')
    # find the last data index in current directory
    i = len(os.listdir(os.getcwd())) + 1
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            horizontal_img = cv2.flip(img, 1)
            cv2.imwrite(str(i) + ".jpg", horizontal_img)
            i += 1

if __name__ == '__main__':
    #detect_edges()
    #rename_img()
    #flip_img()
    pass