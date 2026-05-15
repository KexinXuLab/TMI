from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import gzip
import os
from scipy.stats import pearsonr
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


#张量互信息函数
def extract(matrix):
    m, n = np.shape(matrix)
    Extracting_elements = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            Extracting_elements.append([matrix[i - 1, j - 1], matrix[i - 1, j], matrix[i - 1, j + 1],
                                        matrix[i, j - 1], matrix[i, j], matrix[i, j + 1],
                                        matrix[i + 1, j - 1], matrix[i + 1, j], matrix[i + 1, j + 1]])
    Extracting_elements = np.array(Extracting_elements)
    final_random_variable = np.mean(Extracting_elements, axis=1)
    final_random_variable = (final_random_variable - np.mean(final_random_variable)) / np.std(final_random_variable)
    final_random_variable = np.round(final_random_variable, 0)

    return final_random_variable

def ML(vec1, vec2):
    ML_matrix = np.zeros((len(vec1), len(vec2)))
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            if (vec1[i] == vec2[j]).all():
                ML_matrix[i][j] = 0
            else:
                ML_matrix[i][j] = 1 - normalized_mutual_info_score(vec1[i], vec2[j])

    return ML_matrix

def refind_center(matrix):
    # vec1:31*784, vec2:31*784
    ML_matrix = np.zeros((len(matrix), len(matrix)))  # 10*1000
    for ii in range(len(matrix)):
        for jj in range(len(matrix)):
            ML_matrix[ii][jj] = 1 - normalized_mutual_info_score(matrix[ii], matrix[jj])

    index_num = np.argmin(np.sum(ML_matrix, axis=1))
    return index_num

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

class K_Means_ML(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max = 50):
        I = np.eye(self.n_clusters)
        start_loca = np.random.choice(len(X), self.n_clusters, replace=False)
        centers = X[start_loca]  # 10*784
        for ite in range(iter_max):
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

    # 加载数据（使用您提供的代码）
    data_folder = './MNIST_data'

    x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)

    # 创建500张3和500张5的混合数据集
    photo_count = 1000
    a1 = 4900
    a2 = 4900
    print(a1, a2)
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
    x_train = x_train_gz[label_all]
    y_train = y_train_gz[label_all]

    print(f"混合数据集形状: X={x_train.shape}, y={y_train.shape}")
    print(f"数字3的数量: {np.sum(y_train == 3)}, 数字5的数量: {np.sum(y_train == 5)}")

    data = []
    for i in range(photo_count):
        data_mnist = x_train[i]
        data_mnist = np.reshape(data_mnist, (28, 28))
        data.append(extract(data_mnist))
    data = np.array(data)
    print(data.shape)

    # 聚类类别数
    k_count = 2
    recycle = 100

    ML_count = []
    for m in range(100):
        method1 = K_Means_ML(k_count)
        center = method1.fit(data)
        label1 = method1.predict(data)
        adjusted = accuracy_adjust(y_train, label1)

        print(f"数字3的数量: {np.sum(adjusted == 3)}, 数字5的数量: {np.sum(adjusted == 5)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y_train) / len(y_train)

        ML_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        y_pred = np.array(adjusted)
        y_true = np.array(y_train)
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
        print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy, np.max(ML_count), np.mean(ML_count), np.std(ML_count, ddof=1))

    print('互信息k-means聚类正确率最大值：', np.max(ML_count), np.mean(ML_count), np.std(ML_count, ddof=1))










