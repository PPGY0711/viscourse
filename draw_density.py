# -*- coding:utf-8 -*-
"""
Created on 2020/11/29
@Author: ppgy0711
@Contact: pgy20@mails.tsinghua.edu.cn
@Function: Implement the density Map of Visualization based on MNIST DataSet
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.stats as st


def pre_handle_images(dataSet, saturated=False):
    """
    If the input ndarray.shape == 3 and image pixels are represented using a matrix, flatten the pixel matrix first.
    Prepare data for PCA
    :param images:
    :return: Y with 2 dimensions
    """
    Y = []
    if dataSet.ndim == 3:
        if not saturated:
            for i in range(dataSet.shape[0]):
                image = dataSet[i, :, :]
                # 不能直接把灰度加进去，后面的D算出来太大了导致了exp之后P算出来基本上都是0
                # 这个对每一个像素点的值归一化一下（每一个都除255）
                pixels = image.flatten()/255
                # print(pixels)
                Y.append(pixels)
            Y = np.array(Y)
            return Y
        else:
            for i in range(dataSet.shape[0]):
                image = dataSet[i, :, :]
                # 将像素提升到只有0和255之分
                image = image.flatten()
                image = np.where(image > 0, 255, 0)
                pixels = image.flatten()/255
                # print(pixels)
                Y.append(pixels)
            Y = np.array(Y)
            return Y
    else:
        return dataSet


def tsne(dims=2):
    """
    Using sklearn TSNE package to reduce the data dimension to 2-D
    :param X:
    :param dims:
    :return:
    """
    X = np.load('data/sampled_image.npy')
    # X = pre_handle_images(X)
    X = pre_handle_images(X,True)
    labelSet = np.load('data/sampled_label.npy')
    # X = np.loadtxt('data/mnist2500_X.txt')
    # labelSet = np.loadtxt('data/mnist2500_labels.txt')
    X_embedded = TSNE(n_components=dims).fit_transform(X)
    np.save('sklearn_tsne/mnist1000_X_1.npy', X_embedded)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 20, labelSet)
    # plt.savefig('sklearn_tsne/mnist2500_X_1.png')
    plt.savefig('sklearn_tsne/mnist1000_X_1.png')
    # plt.savefig('sklearn_tsne/mnist1000_X_1.png')
    plt.show()


def kde(data=np.array([])):
    """
    Using different kernel function (6 kinds) to esimate the density function
    :param X:
    :param kernel: kernel function index
    :return:
    """
    # Prepare for data
    # kernel_functions = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    # kernel = kernel_functions[kernel]
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = int(np.min(data[:, 0])-10), int(np.max(data[:, 0])+10)
    ymin, ymax = int(np.min(data[:, 1])-10), int(np.max(data[:, 1])+10)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, 10, cmap='Blues')
    cb = fig.colorbar(cfset)
    # plt.colorbar()
    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    cset = ax.contour(xx, yy, f, 10, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # zz = griddata
    # plt.colorbar(fig, ax=[cset, cfset])
    plt.savefig('sklearn_tsne/mnist%d_X_density_map.png' % (data.shape[0]))
    plt.show()


if __name__ == "__main__":
    # print()
    dataSet = np.load('sklearn_tsne/mnist1000_X_0.npy')
    # print(dataSet)
    kde(dataSet)
    # tsne()
