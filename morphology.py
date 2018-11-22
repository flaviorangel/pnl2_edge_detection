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
            se_j = -1
            do_hit_or_fit = False
            for p in range(-se_center, se_center + 1):
                se_i += 1
                if i + p >= height or i + p < 0:
                    if not hit and erode_border and se[se_i][se_j]:
                        do_hit_or_fit = True
                        break
                    continue
                for q in range(-se_center, se_center + 1):
                    se_j += 1
                    if j + q >= width or j + q < 0:
                        if not hit and erode_border and se[se_i][se_j]:
                            do_hit_or_fit = True
                            break
                        continue
                    if (hit and binary_image[i + p][j + q] == 1) or \
                       (not hit and binary_image[i + p][j + q] == 0):
                        do_hit_or_fit = True
                        break
                if do_hit_or_fit:
                    break
            if do_hit_or_fit:
                if hit:
                    new_image[i][j] = 1
                else:
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
    print(' ')
    print(dilation_or_erosion(test_image, se_3, True))
