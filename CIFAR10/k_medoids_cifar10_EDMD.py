import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import cdist, euclidean
import pickle

#加载cifar-10数据集
def unpickle(file):
    with open("./cifar/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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





if __name__ == '__main__':
    # 调用load_data_gz函数加载mnist数据集
    np.random.seed(42)
    data_folder = './cifar'
    print(os.listdir(data_folder))
    data_batch_1 = unpickle("data_batch_1")  # 打开cifar-10文件的data_batch_1
    cifar_data_1 = data_batch_1[b'data']  # 这里每个字典键的前面都要加上b  (10000, 3072)
    cifar_label_1 = data_batch_1[b'labels']  # (10000,)

    data_batch_2 = unpickle("data_batch_2")  # 打开cifar-10文件的data_batch_2
    cifar_data_2 = data_batch_2[b'data']  # 这里每个字典键的前面都要加上b  (10000, 3072)
    cifar_label_2 = data_batch_2[b'labels']  # (10000,)

    data_batch_3 = unpickle("data_batch_3")  # 打开cifar-10文件的data_batch_3
    cifar_data_3 = data_batch_3[b'data']  # 这里每个字典键的前面都要加上b  (10000, 3072)
    cifar_label_3 = data_batch_3[b'labels']  # (10000,)

    data_batch_4 = unpickle("data_batch_4")  # 打开cifar-10文件的data_batch_4
    cifar_data_4 = data_batch_4[b'data']  # 这里每个字典键的前面都要加上b  (10000, 3072)
    cifar_label_4 = data_batch_4[b'labels']  # (10000,)

    data_batch_5 = unpickle("data_batch_5")  # 打开cifar-10文件的data_batch_5
    cifar_data_5 = data_batch_5[b'data']  # 这里每个字典键的前面都要加上b  (10000, 3072)
    cifar_label_5 = data_batch_5[b'labels']  # (10000,)

    cifar_data_all = []
    cifar_data_all.extend(cifar_data_1)
    cifar_data_all.extend(cifar_data_2)
    cifar_data_all.extend(cifar_data_3)
    cifar_data_all.extend(cifar_data_4)
    cifar_data_all.extend(cifar_data_5)
    cifar_data_all = np.array(cifar_data_all)

    cifar_label_all = cifar_label_1 + cifar_label_2 + cifar_label_3 + cifar_label_4 + cifar_label_5
    cifar_label_all = np.array(cifar_label_all)
    print('全部数据的数据集维度：')
    print(np.shape(cifar_data_all), np.shape(cifar_label_all))  # 五个训练集合数据合并

    photo_count = 1000  # 样本数
    location1 = 4500  # 从哪个位置取
    location2 = 4500  # 从哪个位置取
    print(location1, location2)
    loca_one = np.where(cifar_label_all == 1)
    loca_two = np.where(cifar_label_all == 2)
    print('两类数据的个数：')
    print(len(loca_one[0]), len(loca_two[0]))
    label_one = loca_one[0][location1:location1 + int(photo_count / 2)]  # 取出第一类图像的位置
    label_two = loca_two[0][location2:location2 + int(photo_count / 2)]  # 取出第二类图像的位置

    label_all = []
    for i in range(len(label_one)):
        label_all.append(label_one[i])
        label_all.append(label_two[i])

    X = cifar_data_all[label_all]  # 初始的用于欧氏距离和曼哈顿距离的数据
    y = cifar_label_all[label_all]
    print('训练集个数：')
    print(np.shape(X), np.shape(y))

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

        print(f"数字3的数量: {np.sum(adjusted == 1)}, 数字5的数量: {np.sum(adjusted == 2)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y) / len(y)
        Euc_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.591:
            y_pred = np.array(adjusted)
            y_true = np.array(y)
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 2) & (y_pred == 1))
            TN = np.sum((y_true == 2) & (y_pred == 2))
            FN = np.sum((y_true == 1) & (y_pred == 2))
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

        print(f"数字3的数量: {np.sum(adjusted == 1)}, 数字5的数量: {np.sum(adjusted == 2)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y) / len(y)
        Cit_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.559:
            y_pred = np.array(adjusted)
            y_true = np.array(y)
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 2) & (y_pred == 1))
            TN = np.sum((y_true == 2) & (y_pred == 2))
            FN = np.sum((y_true == 1) & (y_pred == 2))
            total = len(y_true)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            accuracy = (TP + TN) / total if total > 0 else 0
            print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy)

    print('欧氏距离K-Medoids聚类正确率：', Euc_count)
    print('欧氏距离K-Medoids聚类正确率最大值：', np.max(Euc_count), np.mean(Euc_count), np.std(Euc_count, ddof=1))

    print('曼哈顿距离K-Medoids聚类正确率：', Cit_count)
    print('曼哈顿距离K-Medoids聚类正确率最大值：', np.max(Cit_count), np.mean(Cit_count), np.std(Cit_count, ddof=1))




