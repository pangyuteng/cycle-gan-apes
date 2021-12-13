
#from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os


from cyclegan_resnet import CycleGAN

class UpsampleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.img_rowsO = 512
        self.img_colsO = 512
        self.channelsO = 3
        self.img_shapeO = (self.img_rowsO, self.img_colsO, self.channelsO)
        
        # Configure data loader
        self.dataset_name = 'upsample'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (32, 32, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.weight = 10

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_A.summary()
        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_AB.summary()

        self.gan = CycleGAN() # A is ape, B is human.
        self.gan.g_AB.load_weights("saved_model/AB.h5")
        self.gan.g_AB.trainable = False        

        img_sm = Input(shape=self.img_shape)
        img_lg = self.g_AB(img_sm)

        # For the combined model we will only train the generators
        self.d_A.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(img_lg)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_sm],
                              outputs=[ valid_A, img_lg ])
        self.combined.compile(loss=['mse', 'mae'],
                            loss_weights=[  1, self.weight],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Upsampling
        u1 = deconv2d(d0, self.gf*4)
        #u2 = deconv2d(u1, self.gf*2)
        #u3 = deconv2d(u2, self.gf)

        u4 = UpSampling2D(size=2)(u1)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shapeO)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_lg, imgs_sm) in enumerate(self.data_loader.load_batch_for_upsample(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------
                
                fake_B = self.gan.g_AB.predict(imgs_sm)
                fake_A = self.gan.g_BA.predict(fake_B)

                pred_imgs_lg1 = self.g_AB.predict(fake_A)
                pred_imgs_lg2 = self.g_AB.predict(imgs_sm)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_lg, valid)
                dA_loss_fake1 = self.d_A.train_on_batch(pred_imgs_lg1, fake)
                dA_loss_fake2 = self.d_A.train_on_batch(pred_imgs_lg2, fake)
                d_loss = dA_loss_real + dA_loss_fake1 + dA_loss_fake2
                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                
                g_loss = self.combined.train_on_batch([imgs_sm],[valid,imgs_lg])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    self.g_AB.save_weights("saved_model_upsample/AB.h5")
                    self.d_A.save_weights("saved_model_upsample/dA.h5")
                    try:
                        self.gan.g_AB.load_weights("saved_model/AB.h5")
                    except:
                        traceback.print_exc()
          
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1,3

        imgs_lg, imgs_sm = self.data_loader.load_data_for_upsample(domain="A", batch_size=1, is_testing=True)

        fake_B = self.gan.g_AB.predict(imgs_sm)
        fake_A = self.gan.g_BA.predict(fake_B)

        pred_imgs_lg1 = self.g_AB.predict(imgs_sm)
        pred_imgs_lg2 = self.g_AB.predict(fake_A)
        
        gen_imgs = np.concatenate([imgs_lg,pred_imgs_lg1,pred_imgs_lg2])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['original','upsampled','tranformed_upsampled']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(c):
            axs[i].imshow(gen_imgs[cnt])
            axs[i].set_title(titles[i])
            axs[i].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = UpsampleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
