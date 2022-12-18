import cv2
import numpy as np


def score(img1, img2):
    '''
    returns the matching score of images with NCC method
    '''
    img1 = np.ndarray.flatten(img1)
    img2 = np.ndarray.flatten(img2)
    return np.dot(img1 / np.linalg.norm(img1), img2 / np.linalg.norm(img2))


def align(img1, img2, off_x=(-40, 40), off_y=(-40, 40)):
    '''
    An exhaustive search over a window of possible displacements (e.g. [-15,15] pixels), 
    and take the displacement that gives the best matching score. 
    '''
    best_score = -float('inf')
    best_shift = [0, 0]
    
    # loop over all the different displacement permutations
    for i in range(off_x[0], off_x[1] + 1):
        for j in range(off_y[0], off_y[1] + 1):
            temp_score = score(np.roll(img1, (i, j), (0, 1)), img2)
            if temp_score > best_score:
                best_score = temp_score
                best_shift = [i, j]

    # return the best displaced image along with the displacement vector
    return np.roll(img1, best_shift, (0, 1)), best_shift


def split_image_by_height(img):
    '''
    divides the image into 3 equal part according to its height
    '''
    height = img.shape[0] // 3

    b = img[:height]
    g = img[height:height * 2]
    r = img[height * 2:height * 3]
    return b, g, r


def crop_image(img):
    '''
    cropping image %2.5 all sides because of the borders
    '''
    margin = 0.025
    height = img.shape[0]
    width = img.shape[1]
    border_h = int(np.floor(height * margin))
    border_w = int(np.floor(width * margin))

    return img[border_h : height - border_h, border_w : width - border_w]


def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)


def merge_channels(r, g, b):
    return np.dstack((r,g,b))

def histogram_equalisation(img):
    return cv2.equalizeHist(img)

def do_laplacian(img):
    ddepth = cv2.CV_16S
    dst_b = cv2.Laplacian(img, ddepth, ksize=3)
    return dst_b