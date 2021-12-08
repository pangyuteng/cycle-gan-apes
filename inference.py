
import sys
#from cyclegan import CycleGAN
from cyclegan1 import CycleGAN

import numpy as np
from imageio import imwrite

if __name__ == "__main__":

    image_path = sys.argv[1]

    gan = CycleGAN() # A is ape, B is human.
    gan.g_AB.load_weights("saved_model/AB.h5")
    gan.g_BA.load_weights("saved_model/BA.h5")

    img = gan.data_loader.load_img(image_path)
    print(img.shape)
    fake_A = gan.g_BA.predict(img).squeeze()
    fake_A = (0.5 * fake_A + 0.5)*255
    fake_A = fake_A.astype(np.uint8)
    print(fake_A.shape)
    imwrite('inference.png',fake_A)
