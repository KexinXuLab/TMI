import numpy as np
import time
import gzip
import os
import pickle
from scipy.spatial.distance import cdist, euclidean

def unpickle(file):
    with open("./cifar/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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


class KStarMeans:
    """
    k*-means algorithm (Long & Liu, International Journal of Parallel Programming, 2025)
    Paper: "K*-Means: An Efficient Clustering Algorithm with Adaptive Decision Boundaries"

    Algorithm ownership & core idea:
    - Proposed in the above paper (not a rebranding of classical k-means).
    - Core idea: transform the "assign to nearest centroid" step into a hyperplane-based
      classification between clusters, inspired by the perceptron decision boundary concept.
      This reduces repeated point-to-all-centroid distance computations in later iterations.

    Notes on metric:
    - The paper derives the decision-boundary mechanism under the L2 geometry (Euclidean norm).
      We therefore treat the algorithm as Euclidean-based. The `metric` argument is accepted
      for API-compatibility with the original experimental script, but the hyperplane decision
      itself is based on dot products as in Eq.(7)–(13) (paper Sect.3.2).
    """

    def __init__(self, n_clusters: int, metric: str = "euclidean", random_state: int | None = None):
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.random_state = random_state

    # ---------- helpers ----------
    @staticmethod
    def _sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.dot(d, d))

    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # For the first k-means iteration / pruning, we follow the experiment's metric argument.
        return cdist(A, B, metric=self.metric)

    def fit(self, X: np.ndarray, iter_max: int = 200):
        """
        Fit k*-means.

        Implements Algorithm 2 in the paper (Sect.3.2, Algorithm 2; Eq.(7)-(16)).

        Parameters
        ----------
        X : (n_samples, n_features)
        iter_max : maximum number of outer iterations

        Returns
        -------
        centers : (k, n_features) centroids
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        k = self.n_clusters
        if k <= 0 or k > n_samples:
            raise ValueError("n_clusters must be in [1, n_samples].")

        rng = np.random.default_rng(self.random_state)

        # ========== Algorithm 2, line 1: Perform a standard k-means iteration ==========
        # Initialization: randomly select k points as initial centroids (same style as the provided script).
        init_idx = rng.choice(n_samples, k, replace=False)
        centers = X[init_idx].astype(np.float64, copy=True)

        # One Lloyd iteration: assignment then update (paper Sect.2.1, Eq.(2)-(3)).
        D0 = self._dist(X, centers)  # (n, k)
        labels = np.argmin(D0, axis=1)

        # Maintain per-cluster counts and sums to support incremental updates (paper Def.3, Eq.(16)).
        counts = np.zeros(k, dtype=np.int64)
        sums = np.zeros((k, n_features), dtype=np.float64)
        for j in range(k):
            idx = np.where(labels == j)[0]
            counts[j] = len(idx)
            if counts[j] > 0:
                sums[j] = X[idx].mean(axis=0) * counts[j]  # store sum, not mean
            else:
                # empty cluster: re-seed
                ridx = int(rng.integers(0, n_samples))
                labels[ridx] = j
                counts[j] = 1
                sums[j] = X[ridx].astype(np.float64)

        centers = sums / counts[:, None]

        # Main loop: Algorithm 2, line 2
        for _ in range(iter_max):
            labels_prev = labels.copy()

            # ---------- Algorithm 2, lines 3-7: update means & radii, compute inter-centroid distances ----------
            centers = sums / counts[:, None]  # (k,d), Def.3 / Eq.(16) maintained incrementally
            # Ri: radius of each cluster (Eq.(15))
            Ri = np.zeros(k, dtype=np.float64)
            for i in range(k):
                idx_i = np.where(labels == i)[0]
                if len(idx_i) == 0:
                    Ri[i] = 0.0
                    continue
                # Euclidean norm is used in the paper's definition of radius.
                # We compute using squared distances then sqrt to reduce numerical issues.
                dif = X[idx_i] - centers[i]
                Ri[i] = float(np.sqrt(np.max(np.sum(dif * dif, axis=1))))

            # e_{i,j}: inter-centroid edge distance (Eq.(15))
            # Use Euclidean for this edge as per paper.
            # (Even if metric != euclidean, the paper's pruning and hyperplane are Euclidean-geometry based.)
            center_dif = centers[:, None, :] - centers[None, :, :]
            Eij = np.sqrt(np.sum(center_dif * center_dif, axis=2))  # (k,k)

            # Collect all moves first, then apply to keep Eq.(16) bookkeeping consistent.
            move_from = []
            move_to = []

            # ---------- Algorithm 2, lines 8-27: for each cluster, build candidate set, hyperplanes, reassign ----------
            for i in range(k):
                idx_i = np.where(labels == i)[0]
                if len(idx_i) == 0:
                    continue

                # {SC_i}: inter-cluster pruning set (paper Def.2, Eq.(14); Algorithm 2 line 10-12)
                # Condition: Ri > 0.5 * e_{i,j}
                # Exclude j==i.
                candidates = [j for j in range(k) if j != i and Ri[i] > 0.5 * Eij[i, j]]

                if not candidates:
                    continue

                ui = centers[i]

                # Precompute hyperplanes to each candidate cluster (Eq.(7)-(10); Algorithm 2 lines 14-18)
                # For each j:
                #   w_ij = u_j - u_i (Eq.(8))
                #   v_ij = 0.5*(u_i + u_j) (Eq.(9))
                #   b_ij = - w_ij^T v_ij (Eq.(10))
                W = {}
                B = {}
                for j in candidates:
                    uj = centers[j]
                    w = (uj - ui)
                    v = 0.5 * (ui + uj)
                    b = -float(np.dot(w, v))
                    W[j] = w
                    B[j] = b

                # For each point p in C_i (Algorithm 2 line 19)
                for idx in idx_i:
                    p = X[idx].astype(np.float64, copy=False)

                    # intra-cluster pruning (paper Sect.3.2, after Eq.(15); Algorithm 2 line 20-23):
                    # only if ||p - u_i|| > 0.5 * e_{i,j} do we need to evaluate that hyperplane.
                    dp2 = float(np.dot(p - ui, p - ui))
                    dp = float(np.sqrt(dp2))

                    best_val = -np.inf
                    best_j = None

                    for j in candidates:
                        if dp <= 0.5 * Eij[i, j]:
                            continue  # pruned
                        val = float(np.dot(W[j], p) + B[j])  # g_ij(p) by Eq.(7)
                        if val > best_val:
                            best_val = val
                            best_j = j

                    # If max g_ij(p) > 0 => positive class, reassign (Algorithm 2 line 24-26; Eq.(13))
                    if best_j is not None and best_val > 0.0:
                        move_from.append(i)
                        move_to.append(best_j)
                        labels[idx] = best_j

            # ---------- apply incremental updates (Def.3, Eq.(16) idea) ----------
            # We implement Eq.(16) via maintaining per-cluster sums and counts; each moved point updates
            # the two affected clusters' sums and counts (equivalent to adding/removing in Eq.(16)).
            if np.array_equal(labels, labels_prev):
                break

            # Recompute moves precisely (since a point might be set multiple times in loop)
            changed = np.where(labels != labels_prev)[0]
            if len(changed) == 0:
                break

            for idx in changed:
                old = int(labels_prev[idx])
                new = int(labels[idx])
                if old == new:
                    continue
                x = X[idx].astype(np.float64, copy=False)

                # remove from old
                counts[old] -= 1
                sums[old] -= x
                if counts[old] <= 0:
                    # empty cluster handling: re-seed with a random point (keeps algorithm runnable)
                    ridx = int(rng.integers(0, n_samples))
                    labels[ridx] = old
                    counts[old] = 1
                    sums[old] = X[ridx].astype(np.float64)

                # add to new
                counts[new] += 1
                sums[new] += x

        self.centers = (sums / counts[:, None]).astype(np.float64, copy=True)
        self.cluster_labels_ = labels.astype(np.int64, copy=True)
        return self.centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign points to nearest centroid (standard k-means prediction).
        (The provided experiment calls predict(X) on the same X, so this is consistent.)
        """
        X = np.asarray(X)
        D = self._dist(X, self.centers)
        return np.argmin(D, axis=1)


class K_Medoids(object):
    """
    Compatibility wrapper for the original experiment script.

    IMPORTANT:
    - The original file was named k_medoids_mnist.py and the experiment instantiates `K_Medoids`.
      Per the "code replacement principle", we keep the outer API unchanged, but the underlying
      algorithm is replaced by the paper method k*-means.

    Accurate naming:
    - The real implemented method is `KStarMeans` (k*-means) from:
      Long & Liu (2025), International Journal of Parallel Programming.
    """

    def __init__(self, n_clusters, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self._model = KStarMeans(n_clusters=n_clusters, metric=metric)

    def fit(self, X, iter_max=200):
        centers = self._model.fit(X, iter_max=iter_max)
        self.centers = self._model.centers
        self.cluster_labels_ = self._model.cluster_labels_
        return centers

    def predict(self, X):
        return self._model.predict(X)


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
    VERBOSE = True
    vprint = print if VERBOSE else (lambda *a, **k: None)
    # 设置随机种子以确保可重复性
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
    location1 = 4100  # 从哪个位置取
    location2 = 4100  # 从哪个位置取
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

    X = cifar_data_all[label_all]  # 初始的用于欧氏距离和曼哈顿距离的数据
    y = cifar_label_all[label_all]

    print(f"混合数据集形状: X={X.shape}, y={y.shape}")
    print(f"数字3的数量: {np.sum(y == 9)}, 数字5的数量: {np.sum(y == 2)}")


    # 聚类类别数
    k_count = 2
    recycle = 100

    Euc_count = []

    for m in range(recycle):
        kmedoids = K_Medoids(n_clusters=2, metric='euclidean')

        # 训练模型
        centers = kmedoids.fit(X)
        # 预测聚类标签
        labels = kmedoids.predict(X)
        adjusted = accuracy_adjust(y, labels)

        print(f"数字3的数量: {np.sum(adjusted == 9)}, 数字5的数量: {np.sum(adjusted == 2)}")

        # 计算准确率
        accuracy = np.sum(adjusted == y) / len(y)
        Euc_count.append(accuracy)
        print("调整后准确率:", accuracy)  # 应为 1.0

        if accuracy >= 0.1:
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



    Euc_count = np.array(Euc_count)


    # ===== Required report output (0-1 in computation; display as percentage ×100, 1 decimal) =====
    Euc_count = np.array(Euc_count, dtype=np.float64)


    def _fmt(acc_arr: np.ndarray):
        mx = float(np.max(acc_arr))
        mean = float(np.mean(acc_arr))
        sd = float(np.std(acc_arr, ddof=1)) if acc_arr.size > 1 else 0.0
        return mx * 100.0, mean * 100.0, sd * 100.0

    mx_e, mean_e, sd_e = _fmt(Euc_count)


    # Results only:
    print(f"MaxAcc={mx_e:.1f}, MeanAcc±SD={mean_e:.1f}±{sd_e:.1f}")


