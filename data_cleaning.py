import cv2 as cv
import numpy as np
import os
import shutil

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
    data_uncleaned = 'data_uncleaned'
    resize = 'resize'
    assert os.path.exists(data_uncleaned), "Raw data not found"

    copyDirectory(data_uncleaned, resize)
    os.chdir(resize)
    # rename all the images to indices
    i = 1
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            os.rename(image, str(i) + ".jpg")
            i += 1

def flip_img():
    # change directory to data
    os.chdir('../data/Real')
    # clean out the resize folder
    if os.path.exists('../../resize'):
        shutil.rmtree('../../resize')
    # find the last data index in current directory
    i = len(os.listdir(os.getcwd())) + 1
    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            horizontal_img = cv.flip(img, 1)
            cv.imwrite(str(i) + ".jpg", horizontal_img)
            i += 1

def resize_img(new_dimension):
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
            dim = 0, 0
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
            cv.imwrite(image, resized)

def crop_size_check(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return True
    else:
        return False


def crop_img(new_dimension):
    cropped_id = 0
    if os.path.exists('../data/Real'):
        shutil.rmtree('../data/Real')
    os.makedirs('../data/Real')

    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = cv.imread(image)
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
                if not crop_size_check(cropped_img, new_dimension):
                    print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                    cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                cv.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            elif width < height * 1.9:
                crop_start = int(width / 2 - height / 2)
                crop_end = int(crop_start + height)
                cropped_img = img[0:height, crop_start:crop_end]
                if not crop_size_check(cropped_img, new_dimension):
                    print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                    cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                cv.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                cropped_id += 1
            else:
                crop_start = 0
                crop_end = height
                right_limit = img.shape[1]
                while True:
                    cropped_img = img[0:height, crop_start:crop_end]
                    if not crop_size_check(cropped_img, new_dimension):
                        print("{}, {}: {}*{}".format(image, cropped_id, height, width))
                        cv.resize(cropped_img, dsize=(new_dimension,new_dimension), interpolation=cv.INTER_CUBIC)
                    cv.imwrite("../data/Real/" + str(cropped_id) + ".jpg", cropped_img)
                    cropped_id += 1
                    crop_start += int(height * 0.8)
                    crop_end += int(height * 0.8)
                    if crop_end >= right_limit:
                        break


def zero_out(img, edge_only):
    if not edge_only:
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    else:
        # ret, thresh = cv.threshold(img, 255, 255, cv.THRESH_BINARY_INV)
        thresh = img
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
    #img = cv2.imread(imageSource)
    #img = cv2.cvtColor(imageSource,cv2.COLOR_BGR2GRAY).astype(float)

    edge_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    edge_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    edge = np.sqrt(edge_x**2 + edge_y**2)

    #cv.imwrite("edge.jpg", edge)
    return edge

def rgb_to_gray(imageSource):
    img = cv.imread(imageSource)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(float)
    #cv.imwrite("gray.jpg", img)
    return img

def mor_closing(edge_only=False):
    generator_input_dir = '../../input_edges/edges'
    if os.path.exists(generator_input_dir):
        shutil.rmtree(generator_input_dir)
    os.makedirs(generator_input_dir)

    # flipping original data
    for image in os.listdir(os.getcwd()):
        if image.endswith(".jpg"):
            img = rgb_to_gray(image)
            if not edge_only:
                kernel = np.ones((7, 7), np.uint8)
                img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            img = detect_edges(img)
            img = zero_out(img, edge_only)
            img = img.astype('uint8')
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            cv.imwrite(generator_input_dir + "/" + image, img)

if __name__ == '__main__':
    rename_img()
    resize_img(256)
    crop_img(256)
    flip_img()
    mor_closing(False)

