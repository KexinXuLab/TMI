import numpy as np
import time
import gzip
import os
from scipy.spatial.distance import cdist, euclidean


def load_data_gz(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    # 读取每个文件夹的数据
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 784)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 784)

    return x_train, y_train, x_test, y_test


class K_Medoids(object):
    """
    K-Medoids算法实现
    聚类中心始终是实际数据点
    """

    def __init__(self, n_clusters, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric  # 距离度量：'euclidean', 'cityblock', 或自定义

    def fit(self, X, iter_max=200):
        """
        训练K-Medoids模型

        参数:
        X: 输入数据，形状 (n_samples, n_features)
        iter_max: 最大迭代次数

        返回:
        centers: 最终聚类中心（实际数据点）
        """
        n_samples = len(X)
        I = np.eye(self.n_clusters)

        # 1. 随机选择初始聚类中心（实际数据点）
        init_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[init_indices]  # 初始中心是实际数据点
        center_indices = init_indices.copy()  # 记录中心点的索引

        for iteration in range(iter_max):
            prev_center_indices = center_indices.copy()

            # 2. 计算所有点到所有聚类中心的距离
            D = cdist(X, centers, metric=self.metric)  # 形状: (n_samples, n_clusters)

            # 3. 为每个点分配最近的聚类中心
            cluster_labels = np.argmin(D, axis=1)  # 形状: (n_samples,)

            # 4. 转换为one-hot编码
            cluster_mask = I[cluster_labels]  # 形状: (n_samples, n_clusters)

            # 5. 更新聚类中心：为每个簇选择新的中心点（实际数据点）
            new_centers = np.zeros_like(centers)
            new_center_indices = np.zeros(self.n_clusters, dtype=int)

            for k in range(self.n_clusters):
                # 获取属于簇k的所有点的索引
                cluster_k_indices = np.where(cluster_labels == k)[0]

                if len(cluster_k_indices) > 0:
                    # 5a. 计算簇内点到所有点的距离矩阵
                    cluster_points = X[cluster_k_indices]

                    # 如果簇太小，直接选择第一个点作为中心
                    if len(cluster_k_indices) == 1:
                        new_center_idx = cluster_k_indices[0]
                        new_centers[k] = X[new_center_idx]
                        new_center_indices[k] = new_center_idx
                        continue

                    # 5b. 计算簇内所有点之间的成对距离
                    # 方法1: 计算距离矩阵，选择使簇内总距离最小的点
                    pairwise_dist = cdist(cluster_points, cluster_points, metric=self.metric)

                    # 计算每个点到簇内其他点的总距离
                    total_distances = np.sum(pairwise_dist, axis=1)

                    # 选择总距离最小的点作为新中心
                    min_idx_in_cluster = np.argmin(total_distances)
                    new_center_idx = cluster_k_indices[min_idx_in_cluster]

                    # 5c. 更新中心点和索引
                    new_centers[k] = X[new_center_idx]
                    new_center_indices[k] = new_center_idx
                else:
                    # 空簇处理：随机选择一个点作为新中心
                    random_idx = np.random.choice(n_samples)
                    new_centers[k] = X[random_idx]
                    new_center_indices[k] = random_idx

            # 6. 检查是否收敛（中心点是否变化）
            if np.array_equal(prev_center_indices, new_center_indices):
                print(f'收敛于第 {iteration} 次迭代')
                break

            # 7. 更新中心
            centers = new_centers
            center_indices = new_center_indices

        # 保存最终结果
        self.centers = centers
        self.center_indices = center_indices
        self.cluster_labels_ = cluster_labels

        return centers

    def predict(self, X):
        """
        预测新数据点的聚类标签

        参数:
        X: 输入数据，形状 (n_samples, n_features)

        返回:
        labels: 聚类标签，形状 (n_samples,)
        """
        # 计算输入数据点到所有聚类中心的距离
        D = cdist(X, self.centers, metric=self.metric)

        # 返回最近聚类中心的索引
        return np.argmin(D, axis=1)

def accuracy_adjust(label1, label2):  # label1为真实标签，label2为预测标签
    new_label = np.zeros(len(label1))  # 该标签为预测标签调整后与真实标签对应的标签
    remove_repeat_label2 = list(set(label2))
    #print('去重后的预测标签:', remove_repeat_label2)
    for i in range(len(remove_repeat_label2)):
        label_location = [m for m, n in enumerate(label2) if n == remove_repeat_label2[i]]
        #print(remove_repeat_label2[i], label_location)
        location_for_label = []  # 预测标签里面标签remove_repeat_label2[i]对应的位置的真实标签
        for j in range(len(label_location)):
            location_for_label.append(label1[label_location[j]])
        maxlabel = max(location_for_label, key=location_for_label.count)  # 找出出现次数最多的元素
        for mm in range(len(label_location)):
            new_label[label_location[mm]] = maxlabel
    #print(new_label)
    return new_label



# ==================== 主程序 ====================
if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 加载数据（使用您提供的代码）
    data_folder = './MNIST_data'
    x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)

    # 创建500张3和500张5的混合数据集
    photo_count = 1000
    a1 = 2500
    a2 = 2500
    loca_1 = np.where(y_train_gz == 3)
    loca_9 = np.where(y_train_gz == 5)
    print(f"数字3的样本数: {len(loca_1[0])}")
    print(f"数字5的样本数: {len(loca_9[0])}")

    # 选择前500个3和前500个5
    label_1 = loca_1[0][a1:a1 + int(photo_count / 2)]
    label_9 = loca_9[0][a2:a2 + int(photo_count / 2)]

    label_all = []
    for i in range(len(label_1)):
        label_all.append(label_1[i])
        label_all.append(label_9[i])

    # 获取数据
    X = x_train_gz[label_all]
    y = y_train_gz[label_all]

    print(f"混合数据集形状: X={X.shape}, y={y.shape}")
    print(f"数字3的数量: {np.sum(y == 3)}, 数字5的数量: {np.sum(y == 5)}")


    # 聚类类别数
    k_count = 2
    recycle = 100

    Euc_count = []
    Cit_count = []
    for m in range(recycle):
        kmedoids = K_Medoids(n_clusters=2, metric='euclidean')

        # 训练模型
        centers = kmedoids.fit(X)
        # 预测聚类标签
        labels = kmedoids.predict(X)
        adjusted = accuracy_adjust(y, labels)

        print(f"数字3的数量: {np.sum(adjusted == 3)}, 数字5的数量: {np.sum(adjusted == 5)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y) / len(y)
        Euc_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.65:
            y_pred = np.array(adjusted)
            y_true = np.array(y)
            TP = np.sum((y_true == 3) & (y_pred == 3))
            FP = np.sum((y_true == 5) & (y_pred == 3))
            TN = np.sum((y_true == 5) & (y_pred == 5))
            FN = np.sum((y_true == 3) & (y_pred == 5))
            total = len(y_true)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            accuracy = (TP + TN) / total if total > 0 else 0
            print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy)

    for n in range(recycle):
        kmedoids = K_Medoids(n_clusters=2, metric='cityblock')
        # 训练模型
        centers = kmedoids.fit(X)
        # 预测聚类标签
        labels = kmedoids.predict(X)
        adjusted = accuracy_adjust(y, labels)

        print(f"数字3的数量: {np.sum(adjusted == 3)}, 数字5的数量: {np.sum(adjusted == 5)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y) / len(y)
        Cit_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.65:
            y_pred = np.array(adjusted)
            y_true = np.array(y)
            TP = np.sum((y_true == 3) & (y_pred == 3))
            FP = np.sum((y_true == 5) & (y_pred == 3))
            TN = np.sum((y_true == 5) & (y_pred == 5))
            FN = np.sum((y_true == 3) & (y_pred == 5))
            total = len(y_true)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            accuracy = (TP + TN) / total if total > 0 else 0
            print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy)

    Euc_count = np.array(Euc_count)
    Cit_count = np.array(Cit_count)
    print('欧氏距离K-Medoids聚类正确率：', Euc_count)
    print('欧氏距离K-Medoids聚类正确率最大值：', np.max(Euc_count), np.mean(Euc_count), np.std(Euc_count, ddof=1))

    print('曼哈顿距离K-Medoids聚类正确率：', Cit_count)
    print('曼哈顿距离K-Medoids聚类正确率最大值：', np.max(Cit_count), np.mean(Cit_count), np.std(Cit_count, ddof=1))















