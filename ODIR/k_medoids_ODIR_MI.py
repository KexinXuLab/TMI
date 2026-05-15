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

#张量互信息的k-means函数
class K_Means_ML(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max = 50):
        I = np.eye(self.n_clusters)
        start_loca = np.random.choice(len(X), self.n_clusters, replace=False)
        centers = X[start_loca]  # 10*784
        for ite in range(iter_max):
            #print('迭代次数：', ite+1)
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
                break
        self.centers = centers
        return centers

    def predict(self, X):
        D = ML(X, self.centers)
        return np.argmin(D, axis=1)

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

#张量空间信息提取
def extract(tensor):
    m, n, k = np.shape(tensor)
    Extracting_elements1 = []
    Extracting_elements2 = []
    Extracting_elements3 = []
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            Extracting_elements1.append([tensor[i - 1, j - 1, 0], tensor[i - 1, j, 0], tensor[i - 1, j + 1, 0],
                                        tensor[i, j - 1, 0], tensor[i, j, 0], tensor[i, j + 1, 0],
                                        tensor[i + 1, j - 1, 0], tensor[i + 1, j, 0], tensor[i + 1, j + 1, 0]])
            Extracting_elements2.append([tensor[i - 1, j - 1, 1], tensor[i - 1, j, 1], tensor[i - 1, j + 1, 1],
                                         tensor[i, j - 1, 1], tensor[i, j, 1], tensor[i, j + 1, 1],
                                         tensor[i + 1, j - 1, 1], tensor[i + 1, j, 1], tensor[i + 1, j + 1, 1]])
            Extracting_elements3.append([tensor[i - 1, j - 1, 2], tensor[i - 1, j, 2], tensor[i - 1, j + 1, 2],
                                         tensor[i, j - 1, 2], tensor[i, j, 2], tensor[i, j + 1, 2],
                                         tensor[i + 1, j - 1, 2], tensor[i + 1, j, 2], tensor[i + 1, j + 1, 2]])
    Extracting_elements = np.concatenate((Extracting_elements1, Extracting_elements2, Extracting_elements3), axis=0)
    #print(np.shape(Extracting_elements))
    final_random_variable = np.max(Extracting_elements, axis=1)
    final_random_variable = (final_random_variable - np.mean(final_random_variable)) / np.std(final_random_variable)
    final_random_variable = np.round(final_random_variable, 0)

    return final_random_variable









if __name__ == '__main__':
    #np.random.seed(42)
    glaucoma_dir = './eye_data/right_glaucoma'
    normal_dir = './eye_data/right_normal'
    glaucoma_all_images = load_images(glaucoma_dir)
    normal_all_images = load_images(normal_dir)
    print('加载图像的维度：', np.shape(glaucoma_all_images), np.shape(normal_all_images))

    select_count = 0+83+83+83+15
    print(select_count, normal_dir)
    combined_images_all = np.concatenate((glaucoma_all_images, normal_all_images[0 + select_count: 83 + select_count]),
                                         axis=0)
    print('全部训练图像的初始维度：', np.shape(combined_images_all))
    x_train = combined_images_all.reshape(combined_images_all.shape[0], -1)
    print('全部训练图像的最终维度：', np.shape(x_train))

    #data_clear = []  # 用于张量互信息的数据
    #for ia in range(166):
    #    data_clear.append(extract(combined_images_all[ia]))
    #data_clear = np.array(data_clear)
    #print('张量互信息数据维度：', data_clear.shape)

    y_train = [2] * 83 + [9] * 83
    print('全部的标签维度：', np.shape(y_train))

    # 聚类类别数
    k_count = 2
    recycle = 100


    ML_count = []
    for m in range(100):
        method1 = K_Means_ML(k_count)
        center = method1.fit(x_train)
        label1 = method1.predict(x_train)
        adjusted = accuracy_adjust(y_train, label1)

        print(f"数字3的数量: {np.sum(adjusted == 9)}, 数字5的数量: {np.sum(adjusted == 2)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y_train) / len(y_train)

        ML_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

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
        print(TP, FP, TN, FN, precision, recall, f1, specificity, accuracy, np.max(ML_count), np.mean(ML_count),
              np.std(ML_count, ddof=1), m+1)

    print('互信息k-means聚类正确率最大值：', np.max(ML_count), np.mean(ML_count), np.std(ML_count, ddof=1))











