from PIL import Image
import math
import numpy as np


def prepare_image(image_address, binary=False, switch_black_white=False, print_flag=False):
    """Given an image address, return it as an array.

    :param image_address: Location of the image.
    :param binary: if True, return binary array.
    :param switch_black_white: if True, switch 0s and 1s in binary images.
    :param print_flag: if True, will print image as array before returning.
    :return: Numpy array, with size equal to the one of the image.
    """
    im = Image.open(image_address)
    width, height = im.size
    print('this is the image you just loaded:', image_address, im.mode, height, 'x', width)
    if binary:
        im = im.convert('1')
    image_as_array = np.asarray(im, dtype='uint8')
    if binary and switch_black_white:
        image_as_array = (image_as_array - (image_as_array * 2)) + 1
    if print_flag:
        print('')
        print(image_as_array)
    return image_as_array


def save_image(image_as_array, suffix, new_address, binary_image=False, print_flag=True):
    """Saves a numpy array as an image in the given address.
    Address must include name saving type (i.e., png, jpg, gif...).

    :param image_as_array: a numpy array
    :param suffix: string. Any suffix to be added to the name given in address.
    :param new_address: string. Where to save the image. Must include name and type.
    :param binary_image: boolean. If True, the given array has only 0s and 1s.
    :param print_flag: boolean. If True, it will print where the image was saved.
    """
    if binary_image:
        image_as_array = image_as_array * 255
    return_image = Image.fromarray(image_as_array)
    if return_image.mode != 'RGB':
        return_image = return_image.convert('RGB')
    return_image.save(new_address[:-4] + "_" + suffix + new_address[-4:])
    if print_flag:
        print("image saved: " + new_address[:-4] + "_" + suffix + new_address[-4:])


def interpol_lin(image_as_array, location_matrix, transformation_name, height, width, my_image, black=False):
    """Executes a linear interpolation, saving a new image.

    :param image_as_array: The original image represented in x*y array, in which each position is a color.
    :param location_matrix: Each (x,y) gives a location in the original image.
    :param transformation_name: Any string to label the resulting image file.
    :param height: Height of the new image.
    :param width: Width of the new image
    :param my_image: A string that should be the name of the original image file, including its type.
    :param black: bool. If True, positions outside the original image will be filled with the color black.
    """
    print("executing interpolation routine")
    print("this may take a while...")
    new_image = np.copy(image_as_array)
    for i in range(0, height):
        for j in range(0, width):
            if location_matrix[i][j][2] == 1:
                continue
            y = float(location_matrix[i][j][0])
            x = float(location_matrix[i][j][1])
            v0 = int(location_matrix[i][j][0])
            u0 = int(location_matrix[i][j][1])
            if black:
                if u0 < 0 or v0 < 0 or u0 > width-2 or v0 > height-2:
                    new_image[i, j][0] = 0
                    new_image[i, j][1] = 0
                    new_image[i, j][2] = 0
                    continue
            else:
                if u0 < 0:
                    u0 = 0
                if v0 < 0:
                    v0 = 0
                if u0 > width-2:
                    u0 = width-2
                if v0 > height-2:
                    v0 = height-2
            for k in range(0, 3):
                new_image[i, j][k] = int((1 - x + u0)*(1 - y + v0)*image_as_array[v0, u0][k] +
                                         (x - u0)*(1 - y + v0)*image_as_array[v0, u0 + 1][k] +
                                         (1 - x + u0)*(y - v0)*image_as_array[v0 + 1, u0][k] +
                                         (x - u0)*(y - v0)*image_as_array[v0 + 1, u0 + 1][k])
    return_image = Image.fromarray(new_image)
    return_image.save(my_image[:-4] + "_"+transformation_name + my_image[-4:])


def detect_edges_sobel(image_address, threshold=50):
    """Given an image location, detect its edges using Sobel method.

    :param image_address: Location of the image.
    :param threshold: The lower it is, the more parts of the image will be detected as edges.
    """
    print("detecting edges sobel method")
    im = Image.open(image_address)
    im_as_array = prepare_image(image_address)
    width, height = im.size
    edges_image = np.copy(im_as_array)
    for i in range(1, height-1):
        for j in range(1, width-1):
            for k in range(0, 3):
                gx = int(im_as_array[i-1][j-1][k] + 2*im_as_array[i][j-1][k] + im_as_array[i+1][j-1][k])
                gx += -int(im_as_array[i-1][j+1][k] + 2*im_as_array[i][j+1][k] + im_as_array[i+1][j+1][k])
                gy = int(im_as_array[i-1][j-1][k] + 2*im_as_array[i-1][j][k] + im_as_array[i-1][j+1][k])
                gy += -int(im_as_array[i+1][j-1][k] + 2*im_as_array[i+1][j][k] + im_as_array[i+1][j+1][k])
                m = np.sqrt(gx*gx + gy*gy)
                if m > threshold:
                    edges_image[i][j][0] = 0
                    edges_image[i][j][1] = 0
                    edges_image[i][j][2] = 0
                    break
                elif k == 2:
                    edges_image[i][j][0] = 255
                    edges_image[i][j][1] = 255
                    edges_image[i][j][2] = 255
    return_image = Image.fromarray(edges_image)
    return_image.save(image_address[:-4] + "_" + "edges_sobel_" + str(threshold) + image_address[-4:])


def detect_edges_prewitt(image_address, threshold=50):
    """Given an image location, detect its edges using Prewitt method.

    :param image_address: Location of the image.
    :param threshold: The lower it is, the more parts of the image will be detected as edges.
    """
    print("detecting edges prewitt method")
    im = Image.open(image_address)
    im_as_array = prepare_image(image_address)
    width, height = im.size
    edges_image = np.copy(im_as_array)
    for i in range(1, height-1):
        for j in range(1, width-1):
            for k in range(0, 3):
                m1 = int(im_as_array[i+1][j-1][k]) + int(im_as_array[i+1][j][k]) + int(im_as_array[i+1][j+1][k])\
                    - (int(im_as_array[i-1][j-1][k]) + int(im_as_array[i-1][j][k]) + int(im_as_array[i-1][j+1][k]))
                m2 = int(im_as_array[i+1][j][k]) + int(im_as_array[i+1][j+1][k]) + int(im_as_array[i][j+1][k])\
                    - (int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k]) + int(im_as_array[i-1][j][k]))
                m3 = int(im_as_array[i+1][j+1][k]) + int(im_as_array[i][j+1][k]) + int(im_as_array[i-1][j+1][k]) \
                    - (int(im_as_array[i+1][j-1][k]) + int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k]))
                m4 = int(im_as_array[i][j+1][k]) + int(im_as_array[i-1][j+1][k]) + int(im_as_array[i-1][j][k])\
                    - (int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k]) + int(im_as_array[i+1][j][k]))
                m5 = -(int(im_as_array[i+1][j-1][k]) + int(im_as_array[i+1][j][k]) + int(im_as_array[i+1][j+1][k]))\
                    + int(im_as_array[i-1][j-1][k]) + int(im_as_array[i-1][j][k]) + int(im_as_array[i-1][j+1][k])
                m6 = -(int(im_as_array[i+1][j][k]) + int(im_as_array[i+1][j+1][k]) + int(im_as_array[i][j+1][k]))\
                    + int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k]) + int(im_as_array[i-1][j][k])
                m7 = -(int(im_as_array[i+1][j+1][k]) + int(im_as_array[i][j+1][k]) + int(im_as_array[i-1][j+1][k]))\
                    + int(im_as_array[i+1][j-1][k]) + int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k])
                m8 = -(int(im_as_array[i][j+1][k]) + int(im_as_array[i-1][j+1][k]) + int(im_as_array[i-1][j][k])) \
                    + int(im_as_array[i][j-1][k]) + int(im_as_array[i-1][j-1][k]) + int(im_as_array[i+1][j][k])
                m = max(m1, m2, m3, m4, m5, m6, m7, m8)
                if m > threshold:
                    edges_image[i][j][0] = 0
                    edges_image[i][j][1] = 0
                    edges_image[i][j][2] = 0
                    break
                elif k == 2:
                    edges_image[i][j][0] = 255
                    edges_image[i][j][1] = 255
                    edges_image[i][j][2] = 255
    return_image = Image.fromarray(edges_image)
    return_image.save(image_address[:-4] + "_" + "edges_prewitt_" + str(threshold) + image_address[-4:])