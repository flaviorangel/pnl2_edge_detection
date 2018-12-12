import edge_detection
import morphology
import numpy as np
import os


image_antique = "/images/antique.jpg"
image_face = "/images/face.jpg"
image_fog = "/images/fog.jpg"
image_text_gif = "/images/text.gif"
image_text = "/images/text.jpg"
image_text_small = "/images/text_small.png"
se_3x3 = np.ones(shape=(3, 3))
se_3x3[0][0] = 0
se_3x3[0][2] = 0
se_3x3[2][0] = 0
se_3x3[2][2] = 0


if __name__ == "__main__":
    image_address = os.getcwd() + image_text_gif
    image_array = edge_detection.prepare_image(image_address, True, True, print_flag=True)
    edge_detection.save_image(image_array, 'binary', image_address, binary_image=True)
    skeletonized_image = morphology.skeletonization(image_array, se_3x3)
    edge_detection.save_image(skeletonized_image, "skeleton", image_address, binary_image=True)

