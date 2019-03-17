import cv2
import numpy as np
import os
import shutil

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
    os.chdir('data_uncleaned')
    # rename all the images to indices
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            os.rename(image, str(i) + ".jpg")
            # shutil.copyfile(str(i) + ".jpg", "../data/"+ str(i) + ".jpg")
            i += 1

def flip_img():
    # change directory to data
    os.chdir('data/Real')
    # find the last data index in current directory
    i = len(os.listdir(os.getcwd())) + 1
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            horizontal_img = cv2.flip(img, 1)
            cv2.imwrite(str(i) + ".jpg", horizontal_img)
            i += 1

def resize_img(height_pixel):
    # os.chdir('../data')
    os.chdir('data_uncleaned')
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            scale_percent = height_pixel / img.shape[0] # percent of original size
            width = int(img.shape[1] * scale_percent)
            height = height_pixel
            dim = (width, height)
            resized = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image, resized)
            i += 1
        # else:
        #     print("{}".format(i))

def crop_img():
    # os.chdir('data')
    cropped_id = 0
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            height = img.shape[0]
            width = img.shape[1]
            cropped_img = np.empty
            if width < height * 0.8:
                print("Image Removed: ", image)
                os.remove(image)
            elif width < height:
                crop_start = int(height / 2 - width / 2)
                crop_end = int(crop_start + width)
                cropped_img = img[crop_start:crop_end, 0:width]
                cv2.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            elif width < height * 1.9:
                crop_start = int(width / 2 - height / 2)
                crop_end = int(crop_start + height)
                cropped_img = img[0:height, crop_start:crop_end]
                cv2.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            else:
                crop_start = 0
                crop_end = height
                right_limit = img.shape[1]
                while True:
                    cropped_img = img[0:height, crop_start:crop_end]
                    cv2.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                    cropped_id += 1
                    crop_start += int(height * 0.8)
                    crop_end += int(height * 0.8)
                    if crop_end >= right_limit:
                        break
        

if __name__ == '__main__':
    # rename_img()
    # resize_img(224)
    # crop_img()
    flip_img()
    pass