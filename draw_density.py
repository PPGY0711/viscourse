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


def kde(data=np.array([]), filename=""):
    """
    Using different kernel function (gaussian) to esimate the density function
    :param data:
    :return:
    """
    # Prepare for data
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
    # Contour plot
    cset = ax.contour(xx, yy, f, 10, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # zz = griddata
    # plt.colorbar(fig, ax=[cset, cfset])
    # plt.savefig('density/mnist%d_X_density_map.png' % (data.shape[0]))
    plt.savefig(filename)
    plt.show()


def draw_density(X=np.array([]), labels=np.array([]), dims=2, perplexity=30.0, saturated=False):
    """
    Using sklearn TSNE package to reduce the data dimension to 2-D and draw the density Map
    :param X:
    :param dims:
    :return:
    """
    X_embedded = TSNE(n_components=dims, perplexity=perplexity).fit_transform(X)  # Using TSNE provided by sklearn
    # draw density map
    if not saturated:
        if X.shape[0] == 1000:
            np.save('density/mnist1000_X_%d_0.npy' % (int(perplexity)), X_embedded)
            kde(X_embedded, 'density/mnist%d_X_%d_0_density_map.png' % (X_embedded.shape[0], int(perplexity)))
            picname = 'density/mnist%d_X_%d_0.png' % (X_embedded.shape[0], int(perplexity))
        else:
            np.save('density/mnist2500_X_%d_1.npy' % (int(perplexity)), X_embedded)
            kde(X_embedded, 'density/mnist%d_X_%d_1_density_map.png' % (X_embedded.shape[0], int(perplexity)))
            picname = 'density/mnist%d_X_%d_1.png' % (X_embedded.shape[0], int(perplexity))
    else:
        np.save('density/mnist1000_X_%d_1.npy' % (int(perplexity)), X_embedded)
        kde(X_embedded, 'density/mnist%d_X_%d_1_density_map.png' % (X_embedded.shape[0], int(perplexity)))
        picname = 'density/mnist%d_X_%d_1.png' % (X_embedded.shape[0], int(perplexity))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 20, labels)
    plt.savefig(picname)
    plt.show()


def draw_density_map(perplexity=30.0):
    """
    Test function to generate density map with different perplexity
    :param perplexity:
    :return:
    """
    dataSet1 = np.loadtxt('data/mnist2500_X.txt')
    dataSet2 = np.load('data/sampled_image.npy')
    dataSet2 = pre_handle_images(dataSet2)
    dataSet3 = np.load('data/sampled_image.npy')
    dataSet3 = pre_handle_images(dataSet3, True)
    labelSet1 = np.loadtxt('data/mnist2500_labels.txt')
    labelSet2 = np.load('data/sampled_label.npy')
    draw_density(dataSet1, labelSet1, 2, perplexity)
    draw_density(dataSet2, labelSet2, 2, perplexity)
    draw_density(dataSet3, labelSet2, 2, perplexity, True)


if __name__ == "__main__":
    # print()
    draw_density_map(20.0)
    draw_density_map(30.0)
    draw_density_map(40.0)
