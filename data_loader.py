import scipy
from glob import glob
import numpy as np

from imageio import imread
from skimage.transform import resize
import albumentations as A

aug_pipeline = A.Compose([
    A.ShiftScaleRotate(),
])

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        
        #data_type = "train%s" % domain if not is_testing else "test%s" % domain
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        
        if domain == 'A':
            path = glob('/mnt/hd2/data/apebase/ipfs/*')
            descr = 'ape'
        elif domain == 'B':
            #path = glob('/mnt/hd2/data/celeba_gan/img_align_celeba/*')
            path = glob('/mnt/hd2/data/ffhq_dataset/thumbnails128x128/*')
            descr = 'human'
        else:
            raise NotImplementedError()
        
        print(domain, descr,len(path))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            print('original size',img.shape)
            if not is_testing:
                img = resize(img, self.img_res)

                if np.random.random() > 0.5:
                    #img = np.fliplr(img)
                    augmented = aug_pipeline(
                        image=img,
                    )
                    img = augmented['image']
            else:
                img = resize(img, self.img_res)
            imgs.append(img)
        
        imgs = np.array(imgs)/127.5 - 1.
        print(imgs.shape)

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        #path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        #path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        path_A = glob('/mnt/hd2/data/apebase/ipfs/*')
        #path_B = glob('/mnt/hd2/data/celeba_gan/img_align_celeba/*')
        path_B = glob('/mnt/hd2/data/ffhq_dataset/thumbnails128x128/*')
        print(len(path_A),len(path_B))
        
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = resize(img_A, self.img_res)
                img_B = resize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        #img_A = np.fliplr(img_A)
                        #img_B = np.fliplr(img_B)
                        augmented = aug_pipeline(
                            image=img_A,
                        )
                        img_A = augmented['image']
                        augmented = aug_pipeline(
                            image=img_B,
                        )
                        img_B = augmented['image']
                        
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = resize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imread(path, pilmode='RGB').astype(np.float)
