import numpy as np


def dilation(binary_image):
    """Given a binary representation of an image, apply Hit to it.
    The structuring element is a 3x3 box.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black; 1, white.
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


def erosion(binary_image):
    """Given a binary representation of an image, apply Fit to it.
    The structuring element is a 5x5 box.

    :param binary_image: a x*y matrix of 0s and 1s. 0 represents black; 1, white.
    :return: new image with Fit applied.
    """
    height = len(binary_image)
    width = len(binary_image[0])
    new_image = np.copy(binary_image)
    for i in range(0, height):
        for j in range(0, width):
            for k in range(-1, 2):
                if i + k >= height or i + k < 0:
                    continue
                fit = False
                for a in range(-1, 2):
                    if j + a >= width or j + a < 0:
                        continue
                    if binary_image[i + k][j + a] == 0:
                        fit = True
                        break
                if fit:
                    new_image[i][j] = 0
                    break
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
    print(test_image)
    print()
    print(dilation(test_image))
    print()
    print(erosion(test_image))
