import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import cdist, euclidean
import cv2
import pickle

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

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        images.append(img)
    return images

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
    #np.random.seed(42)
    glaucoma_dir = './eye_data/right_glaucoma'
    normal_dir = './eye_data/right_normal'
    glaucoma_all_images = load_images(glaucoma_dir)
    normal_all_images = load_images(normal_dir)
    print('加载图像的维度：', np.shape(glaucoma_all_images), np.shape(normal_all_images))

    select_count = 0
    combined_images_all = np.concatenate((glaucoma_all_images, normal_all_images[0 + select_count: 83 + select_count]),
                                         axis=0)
    print('全部训练图像的初始维度：', np.shape(combined_images_all))
    x_train = combined_images_all.reshape(combined_images_all.shape[0], -1)
    print('全部训练图像的最终维度：', np.shape(x_train))

    y_train = [2] * 83 + [9] * 83
    print('全部的标签维度：', np.shape(y_train))

    # 聚类类别数
    k_count = 2
    recycle = 100

    Euc_count = []
    Cit_count = []
    for m in range(recycle):
        kmedoids = K_Medoids(n_clusters=2, metric='euclidean')

        # 训练模型
        centers = kmedoids.fit(x_train)
        # 预测聚类标签
        labels = kmedoids.predict(x_train)
        adjusted = accuracy_adjust(y_train, labels)

        print(f"数字3的数量: {np.sum(adjusted == 9)}, 数字5的数量: {np.sum(adjusted == 2)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y_train) / len(y_train)
        Euc_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.74:
            y_pred = np.array(adjusted)
            y_true = np.array(y_train)
            TP = np.sum((y_true == 9) & (y_pred == 9))
            FP = np.sum((y_true == 2) & (y_pred == 9))
            TN = np.sum((y_true == 2) & (y_pred == 2))
            FN = np.sum((y_true == 9) & (y_pred == 2))
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
        centers = kmedoids.fit(x_train)
        # 预测聚类标签
        labels = kmedoids.predict(x_train)
        adjusted = accuracy_adjust(y_train, labels)

        print(f"数字3的数量: {np.sum(adjusted == 2)}, 数字5的数量: {np.sum(adjusted == 9)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y_train) / len(y_train)
        Cit_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.63:
            y_pred = np.array(adjusted)
            y_true = np.array(y_train)
            TP = np.sum((y_true == 9) & (y_pred == 9))
            FP = np.sum((y_true == 2) & (y_pred == 9))
            TN = np.sum((y_true == 2) & (y_pred == 2))
            FN = np.sum((y_true == 9) & (y_pred == 2))
            total = len(y_true)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            accuracy = (TP + TN) / total if total > 0 else 0
            print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy)

    #print('欧氏距离k-means聚类正确率：', Euc_count)
    print('欧氏距离K-Medoids聚类正确率最大值：', np.max(Euc_count), np.mean(Euc_count), np.std(Euc_count, ddof=1))

    #print('曼哈顿距离k-means聚类正确率：', Cit_count)
    print('曼哈顿距离K-Medoids聚类正确率最大值：', np.max(Cit_count), np.mean(Cit_count), np.std(Cit_count, ddof=1))









