# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from PIL import ImageFilter


class FourCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1,base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        q1 = self.base_transform1(x)
        k1 = self.base_transform1(x)
        q2 = self.base_transform2(x)
        k2 = self.base_transform2(x)
        return [q1, k1, q2, k2]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
      
class PatchShuffling:
    """Jigsaw style crop
    1st setting n_grid=3, img_size=255, crop_size=64
    2st setting n_grid=2, img_size=255, crop_size=56*2
    3st setting n_grid=3, img_size=224, crop_size=64

    """
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

        self.re_yy = np.reshape(yy * self.crop_size, (n_grid * n_grid,))
        self.re_xx = np.reshape(xx * self.crop_size, (n_grid * n_grid,))

    def __call__(self, img):
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        shuffle(crops)

        shuffling_img = np.zeros([self.crop_size*self.n_grid, self.crop_size*self.n_grid, 3], dtype='uint8')
        for i in range(self.n_grid * self.n_grid):
            shuffling_img[self.re_xx[i]: self.re_xx[i] + self.crop_size, self.re_yy[i]: self.re_yy[i] + self.crop_size] \
                = crops[i]

        return Image.fromarray(shuffling_img)
