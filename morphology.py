import numpy as np


def dilation(binary_image, structuring_element):
    """Given a binary representation of an image, apply Hit to it.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black; 1, white.
    :param structuring_element: a n*n square of 0s and 1s. n must be an odd number.
    :return: new image with Hit applied.
    """
    height = len(binary_image)
    width = len(binary_image[0])
    new_image = np.copy(binary_image)
    for i in range(0, height):
        for j in range(0, width):
            for k in range(-1, 2):
                if i + k >= height or i + k < 0:
                    continue
                hit = False
                for a in range(-1, 2):
                    if j + a >= width or j + a < 0:
                        continue
                    if binary_image[i + k][j + a] == 1:
                        hit = True
                        break
                if hit:
                    new_image[i][j] = 1
                    break
    return new_image


def erosion(binary_image, se, erode_border=False):
    """Given a binary representation of an image, apply Fit to it.
    The structuring element is a 5x5 box.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black; 1, white.
    :param se: the structuring element. A n*n square of 0s and 1s, n must be an odd number.
    :param erode_border: boolean. If True, will apply erosion when the part of the SE is in the border.
    :return: new image with Fit applied.
    """
    se_size = len(se)
    se_center = se_size // 2
    height = len(binary_image)
    width = len(binary_image[0])
    new_image = np.copy(binary_image)
    for i in range(0, height):
        for j in range(0, width):
            se_i = -1
            se_j = -1
            fit = False
            for p in range(-se_center, se_center + 1):
                se_i += 1
                if i + p >= height or i + p < 0:
                    if erode_border and se[se_i][se_j]:
                        fit = True
                        break
                    continue
                for q in range(-se_center, se_center + 1):
                    se_j += 1
                    if j + q >= width or j + q < 0:
                        if erode_border and se[se_i][se_j]:
                            fit = True
                            break
                        continue
                    if binary_image[i + p][j + q] == 0:
                        fit = True
                        break
                if fit:
                    break
            if fit:
                new_image[i][j] = 0
    return new_image


if __name__ == "__main__":
    print("testing routines")
    test_image = np.zeros(shape=(12, 12))
    test_image[1][10] = 1
    for n in range(2, 5):
        test_image[2][n] = 1
        test_image[3][n + 1] = 1
        test_image[10][n + 4] = 1
    for m in range(4, 10):
        for n in range(2, 7):
            test_image[m][n] = 1
    test_image[5][2] = 0
    test_image[5][4] = 0
    test_image[9][3] = 0
    test_image[5][7] = 1
    test_image[5][8] = 1
    test_image[6][7] = 1
    test_image[6][8] = 1
    se_3 = np.ones(shape=(3, 3))
    print(test_image)
    # print(' ')
    # print(dilation(test_image, se_3))
    print(' ')
    print(erosion(test_image, se_3))
