import numpy as np


def dilation_or_erosion(binary_image, se, hit, erode_border=False):
    """Given a binary representation of an image, apply Hit or Fit to its every pixel.
    If Hit is applied, the resulting image is the original image dilated.
    Otherwise, if Fit is applied, the resulting image is the original image eroded.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black (background); 1, white (the objects).
    :param se: the structuring element. A n*n square of 0s and 1s, n must be an odd number.
    :param hit: boolean. If True, Hit will be applied. If False, erosion will be applied.
    :param erode_border: boolean. If True, will apply Fit when the part of the SE is in the border.
    :return: new image with Hit or Fit applied to every pixel.
    """
    se_size = len(se)
    se_center = se_size // 2
    height = len(binary_image)
    width = len(binary_image[0])
    new_image = np.copy(binary_image)
    for i in range(0, height):
        for j in range(0, width):
            se_i = -1
            do_hit_or_fit = False
            for p in range(-se_center, se_center + 1):
                se_i += 1
                se_j = -1
                for q in range(-se_center, se_center + 1):
                    se_j += 1
                    if not se[se_i][se_j]:
                        continue
                    if j + q >= width or j + q < 0 or i + p >= height or i + p < 0:
                        if not hit and erode_border:
                            do_hit_or_fit = True
                            break
                        continue
                    if (hit and binary_image[i + p][j + q]) or \
                       (not hit and not binary_image[i + p, j + q]):
                        do_hit_or_fit = True
                        break
                if do_hit_or_fit:
                    break
            if do_hit_or_fit:
                if hit:
                    new_image[i][j] = 255
                else:
                    new_image[i][j] = 0
    return new_image


def subtract_images(image1, image2):
    """Given 2 binary images of same size, do image1 - image2.
    For every pixel, 1 - 1 = 0, 0 - 0 = 0, 1 - 0 = 1 and 0 - 1 = 0.

    :param image1: a binary image, a x*y matrix of 0s and 1s.
    :param image2: a second binary image, a x*y matrix of 0s and 1s.
    :return: image1 - image2.
    """
    height = len(image1)
    width = len(image1[0])
    new_image = np.copy(image1)
    for i in range(0, height):
        for j in range(0, width):
            if image2[i][j]:
                new_image[i][j] = 0
    return new_image


def add_images(image1, image2):
    """Given 2 binary images of same size, do image1 + image2.
    For every pixel, 1 + 1 = 1, 0 + 0 = 0, 1 + 0 = 1 and 0 + 1 = 1.

    :param image1: a binary image, a x*y matrix of 0s and 1s.
    :param image2: a second binary image, a x*y matrix of 0s and 1s.
    :return: image1 + image2.
    """
    height = len(image1)
    width = len(image1[0])
    new_image = np.copy(image1)
    for i in range(0, height):
        for j in range(0, width):
            if image2[i][j]:
                new_image[i][j] = 255
    return new_image


def skeletonization(binary_image, se):
    """Apply Lantuejoul's skeletonization method to a binary representation of an image.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black (background); 1, white (the objects).
    :param se: the structuring element. A n*n square of 0s and 1s, n must be an odd number.
    :return: new image with skeletonization applied.
    """
    height = len(binary_image)
    width = len(binary_image[0])
    skeletonizated_image = np.zeros(shape=(height, width))
    eroded_image = np.copy(binary_image)
    while True:
        eroded_image = dilation_or_erosion(eroded_image, se, False, erode_border=True)
        opened_image = dilation_or_erosion(eroded_image, se, True)
        subtracted_image = subtract_images(binary_image, opened_image)
        skeletonizated_image = add_images(skeletonizated_image, subtracted_image)
        if not np.count_nonzero(opened_image):
            break
        binary_image = np.copy(eroded_image)
    return skeletonizated_image


if __name__ == "__main__":
    print("testing routines")
    test_image = np.zeros(shape=(12, 12))
    test_image[1][10] = 255
    for n in range(2, 5):
        test_image[2][n] = 255
        test_image[3][n + 1] = 255
        test_image[10][n + 4] = 255
    for m in range(4, 10):
        for n in range(2, 7):
            test_image[m][n] = 255
    test_image[5][2] = 0
    test_image[5][4] = 0
    test_image[9][3] = 0
    test_image[5][7] = 255
    test_image[5][8] = 255
    test_image[6][7] = 255
    test_image[6][8] = 255
    se_3 = np.ones(shape=(3, 3))
    print(test_image)
    print(' ')
    # print(dilation_or_erosion(test_image, se_3, True))
    print(skeletonization(test_image, se_3))
