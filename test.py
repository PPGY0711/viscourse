# -*- coding:utf-8 -*-
import numpy as np

if __name__ == "__main__":
    images = np.load('data/sampled_image.npy')
    labels = np.load('data/sampled_label.npy')
    # test
    print(images.shape, labels.shape)  # (1000, 28, 28) (1000,)
    # print(images.shape[1])  # rank是3，维度3维，第一维数组大小是1000,1000行，矩阵是28*28
    print(labels)
    # plt.show()
