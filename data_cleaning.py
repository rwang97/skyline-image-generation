import cv2 as cv
import numpy as np
import os
import shutil

BASE_DIR = os.getcwd()
REAL_DIR = BASE_DIR + '/data/Real/'
INPUT_EDGE_DIR = BASE_DIR + '/input_edges/edges/'
MOR_EDGE_DIR = BASE_DIR + '/mor_edges/edges/'
RESIZE_DIR = BASE_DIR + '/resize/'
DATA_UNCLEANED_DIR = BASE_DIR + '/data_uncleaned/'
DENOISED_DIR = BASE_DIR + '/denoise/denoise/'

# https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/
def copyDirectory(src, dest):
    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

def rename_img():
    copyDirectory(DATA_UNCLEANED_DIR, RESIZE_DIR)
    os.chdir(RESIZE_DIR)
    # rename all the images to indices
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            os.rename(image, str(i) + ".jpg")
            i += 1

def flip_img(path):
    # change directory to data
    os.chdir(path)
    # find the last data index in current directory
    i = len(os.listdir(os.getcwd())) + 1
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            horizontal_img = cv.flip(img, 1)
            cv.imwrite(str(i) + ".jpg", horizontal_img)
            i += 1

def resize_img(new_dimension, path, option_1, Gray=False):
    # go to specific directory for resizing
    os.chdir(path)
    print("resizing: ", path)
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            dim = (new_dimension, new_dimension)
            if option_1:
                if img.shape[1] > img.shape[0]:  # A wide image
                    scale_percent = new_dimension / img.shape[0]  # percent of original size
                    width = int(img.shape[1] * scale_percent)
                    height = new_dimension
                    dim = (width, height)
                elif img.shape[1] == img.shape[0]:
                    dim = (new_dimension, new_dimension)
                else:
                    scale_percent = new_dimension / img.shape[1]
                    height = int(img.shape[0] * scale_percent)
                    width = new_dimension
                    dim = (width, height)

            resized = cv.resize(img, dsize=dim, interpolation=cv.INTER_CUBIC)
            if Gray == True:
                resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY).astype(float)
            cv.imwrite(image, resized)

def crop_size_check(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return True
    else:
        return False

def crop_img(new_dimension, path, option_1):
    cropped_id = 0
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            height = img.shape[0]
            width = img.shape[1]
            if width < height * 0.8:
                print("Image Removed: ", image)
                os.remove(image)
            elif width < height:
                crop_start = int(height / 2 - width / 2)
                crop_end = int(crop_start + width)
                cropped_img = img[crop_start:crop_end, 0:width]
                if option_1 and not crop_size_check(cropped_img, new_dimension):
                    print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                    cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                cv.imwrite(path + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            elif width < height * 1.9:
                crop_start = int(width / 2 - height / 2)
                crop_end = int(crop_start + height)
                cropped_img = img[0:height, crop_start:crop_end]
                if option_1 and not crop_size_check(cropped_img, new_dimension):
                    print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                    cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                cv.imwrite(path + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            else:
                crop_start = 0
                crop_end = height
                right_limit = img.shape[1]
                while True:
                    cropped_img = img[0:height, crop_start:crop_end]
                    if option_1 and not crop_size_check(cropped_img, new_dimension):
                        print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                        cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                    cv.imwrite(path + str(cropped_id) + ".jpg", cropped_img)
                    cropped_id += 1
                    crop_start += int(height * 0.8)
                    crop_end += int(height * 0.8)
                    if crop_end >= right_limit:
                        break

def zero_out(img, no_morph):
    if no_morph:
        # invert colours
        thresh = 255 - img
    else:
        # apply threshold
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

        # ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        # thresh = 255 - cv.fastNlMeansDenoising(img)

    # ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # img = cv.normalize(img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # Otsu's thresholding
    #ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # adaptive mean
    # thresh4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
    # thresh5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)
    # cv.imwrite("thresh1.jpg", thresh1)
    #cv.imwrite("thresh2.jpg", thresh2)
    # cv.imwrite("thresh3.jpg", thresh3)
    # cv.imwrite("thresh4.jpg", thresh4)
    # cv.imwrite("thresh5.jpg", thresh5)
    return thresh

# https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
def detect_edges(img):
    edge_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    edge_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    edge = np.sqrt(edge_x**2 + edge_y**2)

    return edge

def mor_closing(path, no_morph, Gray):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    print("morphology to: ", path)
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            if no_morph == False:
                kernel = np.ones((7, 7), np.uint8)
                img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # convert to gray before edge detecting
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float)
            img = detect_edges(img)
            img = zero_out(img, no_morph)
            if Gray == False:
                img = img.astype('uint8')
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            cv.imwrite(path + "/" + image, img)

def denoise(path, denoised_path):
    if os.path.exists(denoised_path):
        shutil.rmtree(denoised_path)
    os.makedirs(denoised_path)

    os.chdir(path)

    print("denosing images: ", path)
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            img = np.where(img > 200, 255, img)
            cv.imwrite(denoised_path + "/" + image, img)

if __name__ == '__main__':
    assert os.path.exists(DATA_UNCLEANED_DIR), "Raw data not found"
    option_1 = False
    if option_1: # resize, crop, flip, morph, edge
        rename_img()
        resize_img(new_dimension=256, path=RESIZE_DIR, option_1=option_1, Gray=False)
        crop_img(new_dimension=256, path=REAL_DIR, option_1=option_1)
        flip_img(path=REAL_DIR)
        mor_closing(path=INPUT_EDGE_DIR, no_morph=False, Gray=False)

    else: # crop, flip, (morph) edge, resize
        rename_img()
        crop_img(new_dimension=256, path=REAL_DIR, option_1=option_1)  # crop images into real directory
        flip_img(path=REAL_DIR) # flip images in real directory

        mor_closing(path=INPUT_EDGE_DIR, no_morph=True, Gray=True) # detect edges without morphology, output to input edge directory

        resize_img(new_dimension=256, path=REAL_DIR, option_1=option_1, Gray=False) # go to specific directories for resizing
        # mor_closing(path=MOR_EDGE_DIR, no_morph=False, Gray=True)  # detect edges with morphology, output to input edge directory
        # resize_img(new_dimension=256, path=MOR_EDGE_DIR, option_1=option_1, Gray=True)
        resize_img(new_dimension=256, path=INPUT_EDGE_DIR, option_1=option_1, Gray=False)
        denoise(path=INPUT_EDGE_DIR, denoised_path=DENOISED_DIR)

    # clean out the resize folder
    if os.path.exists(RESIZE_DIR):
        shutil.rmtree(RESIZE_DIR)




