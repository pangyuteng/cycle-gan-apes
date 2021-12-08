
import sys
#from cyclegan import CycleGAN
from cyclegan1 import CycleGAN

import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite

if __name__ == "__main__":
    human_path = sys.argv[1]
    ape_path = sys.argv[2]


    gan = CycleGAN() # A is ape, B is human.
    gan.g_AB.load_weights("saved_model/AB.h5")
    gan.g_BA.load_weights("saved_model/BA.h5")


    imgs_A = gan.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = gan.data_loader.load_img(human_path)

    fake_B = gan.g_AB.predict(imgs_A)
    fake_A = gan.g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = gan.g_BA.predict(fake_B)
    reconstr_B = gan.g_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    r, c = 2, 3
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(ape_path)
    plt.close()

