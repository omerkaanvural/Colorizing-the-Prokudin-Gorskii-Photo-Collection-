from colorize import *
import os
import json


def image_show_df(img):
    '''
    returns five different images (all is merged channels)
    1 -> not aligned image
    2 -> gaussian blurred image
    3 -> blurred and aligned image
    4 -> plus laplacian filtered image
    5 -> plus gamma adjusted image
    '''
    img = cv2.imread(img)
    img = crop_image(img)

    b, g, r = split_image_by_height(img)
    b = b[:,:,0]
    g = g[:,:,0]
    r = r[:,:,0]
    not_aligned_img = merge_channels(b, g, r)

    b = cv2.GaussianBlur(b, (3, 3), 0)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    r = cv2.GaussianBlur(r, (3, 3), 0)
    gaussian_blurred_img = merge_channels(b, g, r)
    
    ag, green_shift = align(g, b)
    ar, red_shift = align(r, b)  
    aligned_img = merge_channels(b, ag, ar)

    aligned_and_laplacian_img = do_laplacian(img)
    gamma_adjusted = adjust_gamma(aligned_and_laplacian_img, 0.8)
    
    shifts = {'green_shift': green_shift, 'red_shift': red_shift}

    return not_aligned_img, gaussian_blurred_img, aligned_img, aligned_and_laplacian_img, gamma_adjusted, shifts


def do_laplacian(img):
    '''
    laplacian filter is applied to cropped and blurred image.
    Then splitted image merged
    '''
    img = crop_image(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ddepth = cv2.CV_8U
    laplacianed_img = cv2.Laplacian(img, ddepth, ksize=1)
    abs_dst = cv2.convertScaleAbs(laplacianed_img)
    img = cv2.subtract(img, abs_dst)
    b, g, r = split_image_by_height(img)

    b = b[:,:,0]
    g = g[:,:,0]
    r = r[:,:,0]

    ab, green_shift = align(b, g)
    ar, red_shift = align(r, g)  
    result = merge_channels(ab, g, ar)
    return result


'''looping image files and 5 image are created 
for each image then save specific file'''
for _, dirs, files in os.walk('images'):
    for image_path in files:
        try:
            folder_path = os.getcwd() + '/results/' + str(image_path.split('.')[0])
            if not os.path.exists(folder_path):
                os.mkdir(folder_path.split('.')[0])
                c = range(6)
                x, y, z, t, q, shifts = image_show_df('images/' + image_path)
                cv2.imwrite(folder_path + '/' + str(c[0]) + '.jpg', x)
                cv2.imwrite(folder_path + '/' + str(c[1]) + '.jpg', y)
                cv2.imwrite(folder_path + '/' + str(c[2]) + '.jpg', z)
                cv2.imwrite(folder_path + '/' + str(c[3]) + '.jpg', t)
                cv2.imwrite(folder_path + '/' + str(c[4]) + '.jpg', q)

                file = open(folder_path + '/shifts.json', 'w')
                json.dump(shifts, file)
            else:
                break

        except FileExistsError:
            print('File exists.')