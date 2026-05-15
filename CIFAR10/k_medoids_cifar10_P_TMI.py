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
    for i in range(len(remove_repeat_label2)):
        label_location = [m for m, n in enumerate(label2) if n == remove_repeat_label2[i]]
        location_for_label = []  # 预测标签里面标签remove_repeat_label2[i]对应的位置的真实标签
        for j in range(len(label_location)):
            location_for_label.append(label1[label_location[j]])
        maxlabel = max(location_for_label, key=location_for_label.count)  # 找出出现次数最多的元素
        for mm in range(len(label_location)):
            new_label[label_location[mm]] = maxlabel
    return new_label

#张量空间信息提取
def extract(tensor):
    m, n, k = np.shape(tensor)
    Extracting_elements = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):

            Extracting_elements.append([tensor[i - 1, j - 1, 0], tensor[i - 1, j, 0], tensor[i - 1, j + 1, 0],
                                        tensor[i, j - 1, 0], tensor[i, j, 0], tensor[i, j + 1, 0],
                                        tensor[i + 1, j - 1, 0], tensor[i + 1, j, 0], tensor[i + 1, j + 1, 0],
                                        tensor[i - 1, j - 1, 1], tensor[i - 1, j, 1], tensor[i - 1, j + 1, 1],
                                        tensor[i, j - 1, 1], tensor[i, j, 1], tensor[i, j + 1, 1],
                                        tensor[i + 1, j - 1, 1], tensor[i + 1, j, 1], tensor[i + 1, j + 1, 1],
                                        tensor[i - 1, j - 1, 2], tensor[i - 1, j, 2], tensor[i - 1, j + 1, 2],
                                        tensor[i, j - 1, 2], tensor[i, j, 2], tensor[i, j + 1, 2],
                                        tensor[i + 1, j - 1, 2], tensor[i + 1, j, 2], tensor[i + 1, j + 1, 2]])
    Extracting_elements = np.array(Extracting_elements)
    middle_element = int((Extracting_elements.shape[1] - 1) / 2)
    coefficient = []
    for i in range(Extracting_elements.shape[1]):
        coefficient.append(pearsonr(Extracting_elements[:, i], Extracting_elements[:, middle_element])[0])
    final_random_variable = np.mean(Extracting_elements * coefficient, axis=1)
    #final_random_variable = (final_random_variable - np.min(final_random_variable))/(np.max(final_random_variable)-np.min(final_random_variable))
    final_random_variable = (final_random_variable - np.mean(final_random_variable)) / np.std(final_random_variable)
    final_random_variable = np.round(final_random_variable, 0)

    return final_random_variable


#互信息距离
def ML(vec1, vec2):
    ML_matrix = np.zeros((len(vec1), len(vec2)))
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            if (vec1[i] == vec2[j]).all():
                ML_matrix[i][j] = 0
            else:
                ML_matrix[i][j] = 1 - normalized_mutual_info_score(vec1[i], vec2[j])

    return ML_matrix

#寻找聚类中心
def refind_center(matrix):
    # vec1:31*784, vec2:31*784
    ML_matrix = np.zeros((len(matrix), len(matrix)))  # 10*1000
    for ii in range(len(matrix)):
        for jj in range(len(matrix)):
            ML_matrix[ii][jj] = 1 - normalized_mutual_info_score(matrix[ii], matrix[jj])

    index_num = np.argmin(np.sum(ML_matrix, axis=1))
    return index_num








class K_Means_ML(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max = 50):
        I = np.eye(self.n_clusters)
        start_loca = np.random.choice(len(X), self.n_clusters, replace=False)
        centers = X[start_loca]  # 10*784
        for ite in range(iter_max):
            print('第 次', ite)
            prev_centers = np.copy(centers)
            D = ML(X, centers) # 1000*10
            cluster_index_num = np.argmin(D, axis=1)  # 1000
            cluster_index = I[cluster_index_num]  # 1000*10
            loca_index = []
            for i in range(self.n_clusters):
                loca = []
                for j in range(len(cluster_index)):
                    if (cluster_index[j] == I[i]).all():
                        loca.append(j)
                X_data = X[loca]
                index = loca[refind_center(X_data)]
                loca_index.append(index)
                centers[i] = X[index]
            if np.allclose(prev_centers, centers):
                print('迭代次数：', ite)
                break
        self.centers = centers
        return centers

    def predict(self, X):
        D = ML(X, self.centers)
        return np.argmin(D, axis=1)


if __name__ == '__main__':

    # 设置随机种子以确保可重复性
    np.random.seed(42)
    # 调用load_data_gz函数加载mnist数据集
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
    print('数据集维度：')
    print(np.shape(cifar_data_all), np.shape(cifar_label_all))  # 五个训练集合数据合并

    photo_count = 1000  # 样本数
    location1 = 4000  # 从哪个位置取
    location2 = 4000  # 从哪个位置取
    print(location1, location2)
    loca_one = np.where(cifar_label_all == 2)
    loca_two = np.where(cifar_label_all == 9)
    print('两类数据的个数：')
    print(len(loca_one[0]), len(loca_two[0]))
    label_one = loca_one[0][location1:location1 + int(photo_count / 2)]  # 取出第一类图像的位置
    label_two = loca_two[0][location2:location2 + int(photo_count / 2)]  # 取出第二类图像的位置

    label_all = []
    for i in range(len(label_one)):
        label_all.append(label_one[i])
        label_all.append(label_two[i])

    x_train = cifar_data_all[label_all]  # 初始的用于欧氏距离和曼哈顿距离的数据
    y_train = cifar_label_all[label_all]
    print(f"混合数据集形状: X={x_train.shape}, y={y_train.shape}")
    print(f"数字3的数量: {np.sum(y_train == 2)}, 数字5的数量: {np.sum(y_train == 9)}")

    data_clear = []  # 用于张量互信息的数据
    x_train = np.reshape(x_train, (photo_count, 32, 32, 3))
    for i in range(photo_count):
        data_clear.append(extract(x_train[i]))
    data_clear = np.array(data_clear)
    print('互信息数据维度：', data_clear.shape)

    # 聚类类别数
    k_count = 2
    recycle = 100

    ML_count = []
    for m in range(50):
        method1 = K_Means_ML(k_count)
        center = method1.fit(data_clear)
        label1 = method1.predict(data_clear)
        adjusted = accuracy_adjust(y_train, label1)

        print(f"数字3的数量: {np.sum(adjusted == 2)}, 数字5的数量: {np.sum(adjusted == 9)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y_train) / len(y_train)

        ML_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

    print('互信息k-means聚类正确率最大值：', np.max(ML_count), np.mean(ML_count))





