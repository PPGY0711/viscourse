# -*- coding:utf-8 -*-
"""
Created on 2020/11/28
@Author: ppgy0711
@Contact: pgy20@mails.tsinghua.edu.cn
@Function: Implement the t-SNE of Visualization based on MNIST DataSet
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image


def draw_images(images, labels, saturated=False):
    """
    save digits images, prepare for interaction
    :param images:
    :param labels:
    :return:
    """
    n = images.shape[0]
    if not saturated:
        for i in range(n):
            if images.ndim == 3:
                image = images[i, :, :]
                pic = Image.fromarray(image).convert("L")
                picname = "img/digit_%d_%d.png" % (i + 1, labels[i])
            else:
                image = images[i, :].reshape(28, 28).T
                image = image*255
                pic = Image.fromarray(image).convert("L")
                picname = "img3/digit_%d_%d.png" % (i+1, labels[i])
            pic.save(picname)
    else:
        for i in range(n):
            if images.ndim == 3:
                image = images[i, :, :]
                image = np.where(image > 0, 255, 0)
                pic = Image.fromarray(image).convert("L")
                picname = "img2/digit_%d_%d.png" % (i + 1, labels[i])
                pic.save(picname)


def draw_all_images():
    draw_images(images, labels, saturated=True)
    draw_images(images, labels)
    draw_images(images2, labels2)


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


def pca(X, n_components):
    """
    Preprocess the dataSet with Principal Component Analysis
    to reduce the calculation complexity and extract the main information
    :param X:dataSet
    :param n_components: the objective low dimension that dataSet is reduced to
    :return:low_dx
    """
    # 1.对原数据集进行零均值化
    means = np.mean(X, axis=0)  # 计算每一列的均值
    mean_values = X - means  # 每一列进行零均值化
    # 2.求协方差
    cov_matrix = np.cov(mean_values, rowvar=False)  # 每一列作为独立变量求协方差，即每一列表示一个属性
    # 3.求协方差矩阵的特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_matrix))
    # 4.将特征值按从大到小排列并选出前r个特征值对应的特征向量
    eig_values_sorted = np.argsort(-eig_values)
    eig_values_sorted = eig_values_sorted[: n_components]
    chosen_eig_vectors = eig_vectors[:, eig_values_sorted]
    # 5.计算降维后的数据集m*r维
    low_dx = mean_values * chosen_eig_vectors
    # 6.将结果还原到m*n维，重构数据，用于调试
    # reconMat = (low_dx * chosen_eig_vectors.T) + means
    # return low_dx, reconMat
    return low_dx


def cal_d(X=np.array([])):
    """
    calculate the Euclidean distances between pairs of data points
    :param X: input data set
    :return: Euclidean distance ndarray
    """
    if X.ndim != 2:
        print("Error: X should be a matrix.")
        return -1
    print("Computing pairwise distances...")
    # 按行加，得到的是1000*1维的列向量，每一行表示每一张图的每个像素点平方之和，即[X0X0T,X1X1T,...XmXmT]T
    X_square_col = np.sum(np.square(X), axis=1)
    # X与X转置的乘积，X_dot_XT(i,j) = XiXjT
    X_dot_XT = np.dot(X, X.T)
    # D(i,j) = XiXiT - 2XiXjT + XjXjT
    D = np.add(np.add(-2*X_dot_XT, X_square_col).T, X_square_col)
    return D


def cal_p(X=np.array([]), tol=1e-5, perplexity=40.0):
    """
    Calculate P-values using binary search according to the distance matirx D
    :param X:
    :param tol:
    :param perplexity:
    :return:
    """
    # 1.计算距离矩阵
    D = cal_d(X)
    # print(D)
    # 2.变量初始化
    (n, d) = X.shape
    Pi = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 200 == 0:
            print("Computing P-values for point %d of %d" % (i, n))
        beta_min = -np.inf
        beta_max = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1: n]))]  # 取Di存储除了D(i,i)之外，所有的D(i,j)
        (H, P) = Hbeta(Di)
        Hdiff = H - logU
        try_time = 0
        # binary search for beta
        while np.abs(Hdiff) > tol and try_time < 50:
            if Hdiff > 0:
                beta_min = beta[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + beta_max)/2.
            else:
                beta_max = beta[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + beta_min) / 2.
            (H, P) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            try_time += 1
        Pi[i, np.concatenate((np.r_[0:i], np.r_[i+1: n]))] = P
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1/beta)))
    return Pi


def Hbeta(Di, beta=1.0):
    """
    Calculate Shannon entropy of xi, i.e Hi(Pi) and Pi={pj|i| i,j<n}
    :param Di: distance matrix without D(i,i), D.ndim = (n-1,)
    :param beta: 1/sigma(i)^2
    :return:(H,P)
    """
    Di = np.array(Di)
    # print(Di)
    Pi = np.exp(-Di.copy()*beta)  # exp(-||xi-xj||^2/2sigma(i)^2)
    # print(Pi.shape)
    sumP = np.sum(Pi)  # Sigma(K!=i) exp(-||xi-xk||^2/2sigma(i)^2)
    # print(sumP.shape)
    Pi = Pi/sumP
    H = np.sum(-Pi * np.log(Pi))
    # print(H)
    return H, Pi


def tsne(X=np.array([]), dims=2, pca_dims=100, perplexity=40.0):
    """
    Using t-SNE on ndarray X to reduce its dimensionality to dims
    :param X: dataset
    :param dims: object dimension
    :param pca_dims: dimension that X has after preprocessed by pca method
    :param perplexity:
    :return:
    """
    # Check before transformation
    if isinstance(dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(dims) != dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables according to the paper by Maaten published in 2008
    X = pca(X, pca_dims).real
    # normalize X again to avoid calculating long distances
    X = (X - np.min(X))/(np.max(X)-np.min(X))
    print(X)
    (n, d) = X.shape
    T = 1000  # iteration times T
    momentum_first = 0.5  # when T < 250
    momentum_second = 0.8  # when T >= 250
    learn_rate = 100
    Y = np.random.randn(n, dims)
    dY = np.zeros((n, dims))
    iY = np.zeros((n, dims))
    gains = np.ones((n, dims))
    min_gain = 0.01

    # reserve for Animation drawing
    # Y_animation = np.zeros((11, n, dims))
    # print(Y_animation.shape)
    # Y_animation[0, :, :] = Y

    # Compute P-values
    P = cal_p(X, 1e-5, perplexity)  # Pj|i
    # print(P)
    P = P + np.transpose(P)
    # print(np.sum(P))
    P = P / np.sum(P)  # 对称化,np.sum(p)=2n，因为每一个x_i对应的概率分布加起来是2(加上转置之后),np.sum(P)一共加了n次
    P = P * 4.  # early exaggeration，这一步来自于论文，最开始的50次迭代使用了夸张
    P = np.maximum(P, 1e-12)  # 将小于e-12的都抹去，为了计算的稳定性

    # Run iterations
    for t in range(T):

        # Compute pairwise affinities(Student t-distribution)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1./(1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q  # pij-qij
        for i in range(n):
            # partial derivative: dC/dYi
            dY[i, :] = 4 * np.sum(np.tile(PQ[:, i] * num[:, i], (dims, 1)).T * (Y[i, :] - Y), 0)
            # dY[i, :] = 4 * np.sum(PQ[i, :] * num[i, :], axis=0)

        # Perform the update
        if t < 250:
            momentum = momentum_first
        else:
            momentum = momentum_second
        #  Jacob(1988)
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - learn_rate * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (t + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (t + 1, C))

        # if (t + 1) % 100 == 0:
        #     Y_animation[int((t + 1)/100), :, :] = Y

        # Stop lying about P-values
        if t == 50:
            P = P / 4.
    # Return solution
    # return Y, Y_animation # for animation drawing
    return Y


def draw_tsne(dataSet, labelSet, dims, pca_dims, perplexity, saturated=False):
    """
    user interface of draw 2-D scatter after applying t-SNE method on high dimension dataSet
    :param dataSet:
    :param labelSet:
    :param dims:
    :param pca_dims:
    :param perplexity:
    :param saturated:
    :return:
    """
    # preprocess the image data(mainly flatten to 1-D)
    Y = pre_handle_images(dataSet, saturated)
    # (Y, Y_animation) = tsne(Y, dims=dims, pca_dims=pca_dims, perplexity=perplexity)  # perplexity 30似乎聚类效果最好

    # t-sne
    Y = tsne(Y, dims=dims, pca_dims=pca_dims, perplexity=perplexity)  # perplexity 30似乎聚类效果最好

    # divide labels into categories
    scatter_array = np.zeros((10, dataSet.shape[0], 3))
    cur_pos = np.zeros(10).astype(int)
    labelSet = labelSet.astype(int)
    # The optimal set of ten distinctive colours from Glasbey(2006)
    colorSet = [(90, 0, 13), (0, 255, 233), (23, 169, 255), (255, 232, 0), (8, 0, 91),
                (4, 255, 4), (0, 0, 255), (0, 79, 0), (255, 21, 205), (255, 0, 0)]
    # divide digits according to label
    for i in range(Y.shape[0]):
        point = Y[i, :]
        label = labelSet[i]
        scatter_array[label, cur_pos[label], :] = np.array([point[0], point[1], i])
        cur_pos[label] = cur_pos[label]+1

    # write result to file in order to analyze the density
    np.save('low_data/points_%d_perplexity_%d_%d.npy' % (dataSet.shape[0], int(perplexity), saturated), Y)
    np.save('low_data/labels_%d_perplexity_%d_%d.npy' % (dataSet.shape[0], int(perplexity), saturated), labelSet)

    traces = []
    for i in range(10):
        if not saturated:
            if Y.shape[0] == 1000:
                hover_filenames = ['img/digit_%d_%d.png' % (scatter_array[i, j, 2], i) for j in
                                   range(scatter_array.shape[1])]
            else:
                hover_filenames = ['img3/digit_%d_%d.png' % (scatter_array[i, j, 2], i) for j in
                                   range(scatter_array.shape[1])]
        else:
            hover_filenames = ['img2/digit_%d_%d.png' % (scatter_array[i, j, 2], i) for j in
                               range(scatter_array.shape[1])]
        trace = go.Scatter(
            x=scatter_array[i, :, 0],
            y=scatter_array[i, :, 1],
            name='digit_%d' % i,
            mode='markers',
            marker={
                'size': 8,
                'color': 'rgb(%d,%d,%d)' % (colorSet[i][0], colorSet[i][1], colorSet[i][2]),
            },
            text=hover_filenames,
            hoverinfo='name+text'
        )
        traces.append(trace)
    layout = go.Layout(
        showlegend=True,
        legend={'font': {'size': 16}, 'x': 1, 'y': 1}
    )
    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig, filename='html/points_%d_perplexity_%d_%d.html' % (dataSet.shape[0], int(perplexity), saturated))
    plt.scatter(Y[:, 0], Y[:, 1], 20, labelSet)
    plt.savefig('png/points_%d_perplexity_%d_%d.png' % (dataSet.shape[0], int(perplexity), saturated))
    plt.show()


if __name__ == "__main__":
    # load file
    images = np.load('data/sampled_image.npy')
    labels = np.load('data/sampled_label.npy')
    images2 = np.loadtxt('data/mnist2500_X.txt')
    labels2 = np.loadtxt('data/mnist2500_labels.txt')
    # print(images2.shape, labels2.shape)
    # draw_all_images()
    draw_tsne(dataSet=images, labelSet=labels, dims=2, pca_dims=40, perplexity=50.0)
    draw_tsne(dataSet=images, labelSet=labels, dims=2, pca_dims=40, perplexity=50.0, saturated=True)
    draw_tsne(dataSet=images2, labelSet=labels2, dims=2, pca_dims=40, perplexity=50.0)
