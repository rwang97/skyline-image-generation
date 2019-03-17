import cv2
import numpy as np
import os
import shutil

def rename_img():
    # change directory to data
    assert os.path.exists('data_uncleaned'), "Raw data not found"
    os.chdir('data_uncleaned')
    # rename all the images to indices
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            os.rename(image, str(i) + ".jpg")
            i += 1

def flip_img():
    # change directory to data
    os.chdir('../data/Real')
    # find the last data index in current directory
    i = len(os.listdir(os.getcwd())) + 1
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            horizontal_img = cv2.flip(img, 1)
            cv2.imwrite(str(i) + ".jpg", horizontal_img)
            i += 1

def resize_img(new_dimension):
    # os.chdir('data_uncleaned')
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv2.imread(image)
            dim = 0, 0
            if img.shape[1] > img.shape[0]:  # A wide image
                scale_percent = new_dimension / img.shape[0]  # percent of original size
                width = int(img.shape[1] * scale_percent)
                height = new_dimension
                dim = (width, height)
            else:
                scale_percent = new_dimension / img.shape[1]
                height = int(img.shape[0] * scale_percent)
                width = new_dimension
                dim = (width, height)
            resized = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image, resized)

def crop_img():
    cropped_id = 0
    if os.path.exists('../data/Real'):
        shutil.rmtree('../data/Real')
    os.makedirs('../data/Real')

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
    rename_img()
    resize_img(224)
    crop_img()
    flip_img()
