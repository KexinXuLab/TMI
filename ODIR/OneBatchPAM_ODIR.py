import numpy as np
import gzip
import os
import pickle
import cv2
from sklearn.metrics import pairwise_distances
def unpickle(file):
    with open("./cifar/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_gz(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
    ]

    paths = [os.path.join(data_folder, f) for f in files]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 784)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 784)

    return x_train, y_train, x_test, y_test


def accuracy_adjust(label1, label2):
    """Map cluster ids to true labels by majority vote."""
    new_label = np.zeros(len(label1))
    remove_repeat_label2 = list(set(label2))
    for i in range(len(remove_repeat_label2)):
        label_location = [m for m, n in enumerate(label2) if n == remove_repeat_label2[i]]
        location_for_label = []
        for j in range(len(label_location)):
            location_for_label.append(label1[label_location[j]])
        maxlabel = max(location_for_label, key=location_for_label.count)
        for mm in range(len(label_location)):
            new_label[label_location[mm]] = maxlabel
    return new_label

# ============================================================
# Exp7 core algorithm: OneBatchPAM (paper: "OneBatchPAM: A Fast and Frugal K-Medoids Algorithm")
# The following two functions are a *minimal* transcription of the authors'
# public code path used in `onebatch.py`, except that the original C-extension
# `pam.swap_eager` is replaced by a NumPy implementation with the same interface.
# ============================================================

def swap_eager_numpy(Dist, medoids_init, K, n_swap, N, B, tol_init):
    """Eager-swap local search on the 1-batch estimated objective (NumPy).

    Dist: (N, B) float32, distances from any candidate (rows) to each batch point (cols).
    medoids_init: (K,) int32, initial medoid indices in [0, N).
    """
    Dist = np.asarray(Dist, dtype=np.float32, order="C")
    medoids = np.asarray(medoids_init, dtype=np.int32).copy()

    if K <= 0:
        return medoids

    N0, B0 = Dist.shape
    if N0 != N or B0 != B:
        N, B = N0, B0

    # ensure K unique medoids
    medoids = np.unique(medoids)
    if medoids.size < K:
        pool = np.setdiff1d(np.arange(N, dtype=np.int32), medoids, assume_unique=False)
        extra = np.random.choice(pool, K - medoids.size, replace=False).astype(np.int32)
        medoids = np.concatenate([medoids, extra])
    elif medoids.size > K:
        medoids = medoids[:K]

    tol = float(tol_init)

    def _nearest_second(meds):
        Dm = Dist[meds, :]  # (K, B)
        order = np.argsort(Dm, axis=0)
        nearest_pos = order[0, :]
        second_pos = order[1, :] if K > 1 else order[0, :]
        cols = np.arange(B)
        min_d = Dm[nearest_pos, cols]
        sec_d = Dm[second_pos, cols]
        return min_d, sec_d, nearest_pos

    for _ in range(int(n_swap)):
        min_d, sec_d, nearest_pos = _nearest_second(medoids)

        best_gain = tol
        best_out_pos = -1
        best_in = -1

        is_medoid = np.zeros(N, dtype=bool)
        is_medoid[medoids] = True

        # loop over which medoid to remove
        for out_pos in range(K):
            assigned = (nearest_pos == out_pos)
            not_assigned = ~assigned

            # scan all candidates to add
            for i in range(N):
                if is_medoid[i]:
                    continue
                di = Dist[i, :]

                # points not assigned to out_pos
                cur_na = min_d[not_assigned]
                new_na = np.minimum(cur_na, di[not_assigned])
                gain_na = float(np.sum(cur_na - new_na))

                # points assigned to out_pos
                cur_a = min_d[assigned]
                base_a = sec_d[assigned]
                new_a = np.minimum(base_a, di[assigned])
                gain_a = float(np.sum(cur_a - new_a))

                gain = gain_na + gain_a
                if gain > best_gain:
                    best_gain = gain
                    best_out_pos = out_pos
                    best_in = i

        if best_out_pos < 0:
            break

        medoids[best_out_pos] = best_in
        # keep unique length K
        medoids = np.unique(medoids)
        if medoids.size < K:
            pool = np.setdiff1d(np.arange(N, dtype=np.int32), medoids, assume_unique=False)
            extra = np.random.choice(pool, K - medoids.size, replace=False).astype(np.int32)
            medoids = np.concatenate([medoids, extra])
        elif medoids.size > K:
            medoids = medoids[:K]

    return medoids


def one_batch_pam_numpy(X, K=1, distance="euclidean", batch_size=1000, verbose=0, weight="debias"):
    """Authors' OneBatchPAM entry point, keeping the same signature as onebatch.one_batch_pam."""
    N = X.shape[0]

    if batch_size > N:
        batch_size = N

    if weight == "lwcs":
        x_mean = X.mean(0)
        dist_to_mean = pairwise_distances(X, x_mean.reshape(1, -1), metric=distance).ravel()
        probas = 0.5 * (1 / N) + 0.5 * dist_to_mean / dist_to_mean.sum()
        probas /= probas.sum()
        batch_indexes = np.random.choice(N, batch_size, replace=False, p=probas)
        Dist = pairwise_distances(X, X[batch_indexes], metric=distance)
        Dist = Dist.astype(np.float32)
        Dist /= np.float32(Dist.max() + 1e-12)
        sample_weight = probas[batch_indexes]
        sample_weight /= sample_weight.mean()
        sample_weight = sample_weight.astype(np.float32)
        Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
        Dist *= sample_weight
    else:
        batch_indexes = np.random.choice(N, batch_size, replace=False)
        Dist = pairwise_distances(X, X[batch_indexes], metric=distance)
        Dist = Dist.astype(np.float32)
        Dist /= np.float32(Dist.max() + 1e-12)

        if weight == "debias":
            Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
        elif weight == "nniw":
            sample_weight = np.zeros(batch_size, dtype=np.float32)
            unique, counts = np.unique(Dist.argmin(1).ravel(), return_counts=True)
            sample_weight[unique] = counts
            sample_weight /= sample_weight.mean()
            sample_weight = sample_weight.astype(np.float32)
            Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
            Dist *= sample_weight
        else:
            pass

    medoids = np.random.choice(N, K, replace=False).astype(np.int32)

    new_medoids = swap_eager_numpy(Dist, medoids, K, 100, N, batch_size, np.float32(1e-6))
    return new_medoids


class OneBatchPAMKMedoids:
    """K-medoids clustering wrapper using Exp7 OneBatchPAM core algorithm."""

    def __init__(self, n_clusters: int, metric: str = "euclidean", batch_size="auto", weight="debias"):
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.batch_size = batch_size
        self.weight = weight

    def fit(self, X):
        X = np.asarray(X)
        n = X.shape[0]

        # paper suggests m = O(log n); authors' wrapper uses: int(100 * log(n*K))
        if self.batch_size == "auto":
            bs = int(100 * np.log(max(2, n * self.n_clusters)))
        else:
            bs = int(self.batch_size)
        bs = min(n, max(self.n_clusters, bs))

        # metric mapping for sklearn
        dist = self.metric
        if dist == "cityblock":
            dist = "manhattan"

        medoid_indices = one_batch_pam_numpy(
            X=X,
            K=self.n_clusters,
            distance=dist,
            batch_size=bs,
            verbose=0,
            weight=self.weight,
        )

        self.center_indices = np.asarray(medoid_indices, dtype=int)
        self.centers = X[self.center_indices]
        self.cluster_labels_ = self.predict(X)
        return self.centers

    def predict(self, X):
        X = np.asarray(X)
        dist = self.metric
        if dist == "cityblock":
            dist = "manhattan"
        D = pairwise_distances(X, self.centers, metric=dist)
        return np.argmin(D, axis=1)

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        images.append(img)
    return images
# =========================================================
# Main (kept compatible with your current experiment code)
# =========================================================
if __name__ == '__main__':
    np.random.seed(42)

    # np.random.seed(42)
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

    X = x_train  # 初始的用于欧氏距离和曼哈顿距离的数据
    y = y_train

    print(f"混合数据集形状: X={X.shape}, y={np.shape(y)}")
    print(f"数字3的数量: {np.sum(y == 9)}, 数字5的数量: {np.sum(y == 2)}")

    recycle = 100

    for metric_name in ["cityblock"]:
        acc_list = []
        for _ in range(recycle):
            kmedoids = OneBatchPAMKMedoids(n_clusters=2, metric=metric_name, batch_size="auto", weight="debias")
            kmedoids.fit(X)
            labels = kmedoids.predict(X)
            adjusted = accuracy_adjust(y, labels)

            acc = np.sum(adjusted == y) / len(y)
            print("调整后准确率:", acc)  # 应为 1.0
            acc_list.append(acc)

            y_pred = np.array(adjusted)
            y_true = np.array(y)
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

        acc_array = np.array(acc_list, dtype=np.float64)
        # accuracy -> percentage
        acc_arr = acc_array * 100.0
        print(f"{metric_name} OneBatchPAM 聚类正确率：", acc_arr)
        print(f"{metric_name} OneBatchPAM 聚类正确率统计：max={acc_arr.max():.1f}, Mean ± SD={acc_arr.mean():.1f}±{acc_arr.std(ddof=1):.1f}")
