import numpy as np
import math
import time
import gzip
import os
from scipy.spatial.distance import cdist, euclidean
import pickle
import cv2


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


class K_Sil_KMeans(object):
    """
    K-Sil: Silhouette-Guided Instance-Weighted k-means (Semoglou et al., 2025)

    真实算法归属：
        - 论文：Silhouette-Guided Instance-Weighted k-means（arXiv:2506.12878v1, 2025）
        - 方法名：K-Sil
        - 核心思想：使用每个样本的 silhouette 分数指导“实例加权”的质心更新，
          强化高置信（高 silhouette）点，抑制边界/噪声点，从而提升簇内紧凑与簇间分离。

    重要：本类严格按论文 Appendix A.1 的 K-Sil 算法流程实现，并在关键步骤标注对应公式。
    """

    def __init__(
            self,
            n_clusters,
            metric='euclidean',
            init_method='kmeans++',  # 论文 §3 Initialization：random 或 k-means++
            objective='micro',  # 论文 Eq.(2)：'micro'(Sm) / 'macro'(SM) / 'hybrid'(alpha Sm + (1-alpha) SM)
            alpha=0.5,  # hybrid 组合系数
            weighting_scheme='power',  # 论文 §3.1：'power'(Eq.5) / 'exponential'(Eq.6)
            p=10.0,  # 论文 §3.2：weight-sensitivity parameter p > 0
            epsilon=1e-12,  # 论文 Eq.(5) 数值稳定项 ϵ（论文描述为 small constant）
            tau=1e-4,  # 论文 Appendix A.1：centroid movement threshold τ
            use_approx_silhouette=False,  # 论文 Eq.(3)-(4)：是否使用近似 silhouette
            sampling_size=None,  # 论文 §3.1 / Appendix A.1：Sampling size (m); None 表示不采样
            sampling_with_replacement=False,  # 采样是否放回（论文未强制；默认不放回）
            random_state=None
    ):
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.init_method = init_method
        self.objective = objective
        self.alpha = float(alpha)
        self.weighting_scheme = weighting_scheme
        self.p = float(p)
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.use_approx_silhouette = bool(use_approx_silhouette)
        self.sampling_size = None if sampling_size is None else int(sampling_size)
        self.sampling_with_replacement = bool(sampling_with_replacement)
        self.random_state = random_state

        # outputs
        self.centers = None
        self.cluster_labels_ = None

    # -----------------------------
    # Initialization (§3, Appendix A.1 first line)
    # -----------------------------
    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        n_samples = X.shape[0]

        if self.init_method == 'random':
            idx = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[idx].astype(np.float64, copy=True)

        # k-means++ (Arthur & Vassilvitskii, 2007) — 论文 §3 Initialization
        if self.init_method != 'kmeans++':
            raise ValueError("init_method must be 'random' or 'kmeans++'")

        centers = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
        # choose first center uniformly
        first_idx = rng.randint(0, n_samples)
        centers[0] = X[first_idx]

        # distances to nearest chosen center
        dist_sq = cdist(X, centers[[0]], metric=self.metric).reshape(-1) ** 2
        for c in range(1, self.n_clusters):
            probs = dist_sq / (dist_sq.sum() + 1e-30)
            next_idx = rng.choice(n_samples, p=probs)
            centers[c] = X[next_idx]
            new_dist_sq = cdist(X, centers[[c]], metric=self.metric).reshape(-1) ** 2
            dist_sq = np.minimum(dist_sq, new_dist_sq)

        return centers

    # -----------------------------
    # Silhouette scores — exact Eq.(1) or approx Eq.(3)-(4)
    # -----------------------------
    def _silhouette_exact(self, X, labels):
        """
        Exact silhouette per point, per Eq.(1):
            s(xi) = (b(xi) - a(xi)) / max(a(xi), b(xi))
        where:
            a(xi) = avg distance to points in same cluster
            b(xi) = min over other clusters of avg distance to that cluster
        """
        n = X.shape[0]
        k = self.n_clusters
        # precompute full pairwise distances for exact silhouette (n=1000 is OK)
        D = cdist(X, X, metric=self.metric).astype(np.float64)

        sil = np.zeros(n, dtype=np.float64)
        for i in range(n):
            ci = labels[i]
            in_same = (labels == ci)
            # a(xi): average intra-cluster distance excluding itself
            same_idx = np.where(in_same)[0]
            if same_idx.size <= 1:
                # singleton cluster → silhouette defined as 0
                sil[i] = 0.0
                continue
            a = (D[i, same_idx].sum() - 0.0) / (same_idx.size - 1)  # D[i,i]=0
            # b(xi): min average distance to other clusters
            b = np.inf
            for cj in range(k):
                if cj == ci:
                    continue
                other_idx = np.where(labels == cj)[0]
                if other_idx.size == 0:
                    continue
                b = min(b, D[i, other_idx].mean())
            if not np.isfinite(b):
                sil[i] = 0.0
                continue
            denom = max(a, b)
            sil[i] = 0.0 if denom <= 0 else (b - a) / denom
        return sil

    def _silhouette_approx(self, X, labels, centroids):
        """
        Approx silhouette per point per Eq.(3)-(4) in §3.1.
        Uses cluster sizes, centroids and within-cluster sum of squares (SS).

        a~(xi) = sqrt( (|Cj|*||xi-μj||^2 + SS_Cj) / (|Cj|-1) )
        b~(xi) = min_{h!=j} sqrt( ||xi-μh||^2 + SS_Ch/|Ch| )
        s~(xi) = (b~ - a~) / max(a~, b~)
        """
        n = X.shape[0]
        k = self.n_clusters

        # cluster stats
        sizes = np.zeros(k, dtype=np.int64)
        ss = np.zeros(k, dtype=np.float64)
        for j in range(k):
            idx = np.where(labels == j)[0]
            sizes[j] = idx.size
            if idx.size > 0:
                dif = X[idx].astype(np.float64) - centroids[j]
                ss[j] = np.sum(dif * dif)  # SS_Cj = sum ||x-μj||^2

        sil = np.zeros(n, dtype=np.float64)

        # precompute squared distances to centroids
        dc = cdist(X, centroids, metric=self.metric).astype(np.float64)
        # if metric is euclidean, cdist gives euclidean; square it.
        # for other metrics, paper is L2-based; we follow squared value as proxy for Eq.(3).
        dc2 = dc * dc

        for i in range(n):
            j = labels[i]
            if sizes[j] <= 1:
                sil[i] = 0.0
                continue

            # Eq.(3): a~(xi)
            a2 = (sizes[j] * dc2[i, j] + ss[j]) / max(sizes[j] - 1, 1)
            a = np.sqrt(max(a2, 0.0))

            # Eq.(3): b~(xi)
            b = np.inf
            for h in range(k):
                if h == j or sizes[h] == 0:
                    continue
                b2 = dc2[i, h] + ss[h] / max(sizes[h], 1)
                b = min(b, np.sqrt(max(b2, 0.0)))

            if not np.isfinite(b):
                sil[i] = 0.0
                continue
            denom = max(a, b)
            sil[i] = 0.0 if denom <= 0 else (b - a) / denom  # Eq.(4)
        return sil

    # -----------------------------
    # Silhouette aggregation objective — Eq.(2)
    # -----------------------------
    def _aggregate_objective(self, sil, labels):
        if self.objective == 'micro':
            # Sm = (1/n) sum s(xi)  — Eq.(2)
            return float(np.mean(sil))
        if self.objective == 'macro':
            # SM = (1/k) sum_j mean_{xi in Cj} s(xi) — Eq.(2)
            vals = []
            for j in range(self.n_clusters):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    continue
                vals.append(float(np.mean(sil[idx])))
            return float(np.mean(vals)) if vals else -1.0
        if self.objective == 'hybrid':
            Sm = float(np.mean(sil))
            # macro part
            vals = []
            for j in range(self.n_clusters):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    continue
                vals.append(float(np.mean(sil[idx])))
            SM = float(np.mean(vals)) if vals else -1.0
            return self.alpha * Sm + (1.0 - self.alpha) * SM
        raise ValueError("objective must be 'micro', 'macro', or 'hybrid'")

    # -----------------------------
    # Weighting schemes — Eq.(5) and Eq.(6)
    # -----------------------------
    def _weights_power(self, sil_cluster):
        """
        Power weighting scheme — Eq.(5)
        w_i = [ (s_i - s_min + eps) / median(s_h - s_min + eps) ]^p
        """
        smin = float(np.min(sil_cluster))
        adj = (sil_cluster - smin) + self.epsilon
        med = float(np.median(adj)) if adj.size > 0 else 1.0
        med = med if med > 0 else self.epsilon
        w = (adj / med) ** self.p
        return w

    def _dense_rank_desc(self, arr):
        """
        Descending dense rank:
          - higher score → smaller rank (1 is best)
          - ties share same rank
        """
        # unique sorted descending
        uniq = np.unique(arr)[::-1]
        rank_map = {v: i + 1 for i, v in enumerate(uniq)}
        return np.array([rank_map[v] for v in arr], dtype=np.float64)

    def _weights_exponential(self, sil_cluster):
        """
        Exponential weighting scheme — Eq.(6)
        w_i = exp( -p * ( rank(s_i) - median(rank(s_h)) ) / (rank(s_min)/2) )
        """
        ranks = self._dense_rank_desc(sil_cluster)
        med_rank = float(np.median(ranks)) if ranks.size > 0 else 1.0
        max_rank = float(np.max(ranks)) if ranks.size > 0 else 1.0  # rank(s_min)
        denom = max_rank / 2.0
        denom = denom if denom > 0 else 1.0
        w = np.exp(-self.p * ((ranks - med_rank) / denom))
        return w

    # -----------------------------
    # Objective-aligned sampling (§3.1, Appendix A.1 "Sampling size")
    #   - micro objective Sm: uniform sampling over X (preserves proportional cluster representation)
    #   - macro objective SM: per-cluster balanced sampling (equal points per cluster)
    #   - hybrid: default to micro-aligned uniform sampling (论文只明确 macro/micro 对齐；hybrid 采用 micro 对齐更稳定)
    # -----------------------------
    def _sample_indices_per_cluster(self, labels):
        if self.sampling_size is None:
            return [np.where(labels == j)[0] for j in range(self.n_clusters)]

        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        n = labels.shape[0]
        m = min(int(self.sampling_size), n)

        if self.objective == 'macro':
            # balanced sampling: sample roughly m/k points per cluster
            m_per = max(1, int(math.ceil(m / float(self.n_clusters))))
            per = []
            for j in range(self.n_clusters):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    per.append(idx)
                    continue
                take = min(idx.size, m_per) if not self.sampling_with_replacement else m_per
                chosen = rng.choice(idx, size=take, replace=self.sampling_with_replacement)
                per.append(np.asarray(chosen, dtype=np.int64))
            return per

        # micro / hybrid: uniform sampling over all points, then split by cluster
        chosen_all = rng.choice(np.arange(n, dtype=np.int64), size=m, replace=False)  # uniform, no replacement
        chosen_set = set(chosen_all.tolist())
        per = []
        for j in range(self.n_clusters):
            idx = np.where(labels == j)[0]
            if idx.size == 0:
                per.append(idx)
                continue
            per.append(np.asarray([ii for ii in idx.tolist() if ii in chosen_set], dtype=np.int64))
        return per

    def _compute_weights(self, sil, labels):
        w = np.ones_like(sil, dtype=np.float64)
        for j in range(self.n_clusters):
            idx = np.where(labels == j)[0]
            if idx.size == 0:
                continue
            sil_c = sil[idx]
            if self.weighting_scheme == 'power':
                w[idx] = self._weights_power(sil_c)
            elif self.weighting_scheme == 'exponential':
                w[idx] = self._weights_exponential(sil_c)
            else:
                raise ValueError("weighting_scheme must be 'power' or 'exponential'")
        return w

    # -----------------------------
    # Centroid update — Eq.(7) + empty-cluster reinit — Eq.(8)
    # -----------------------------
    def _update_centroids(self, X, labels, weights, centroids_prev, sampled_per_cluster=None, weight_by_index=None):
        k = self.n_clusters
        d = X.shape[1]
        centroids = np.zeros((k, d), dtype=np.float64)
        sizes = np.zeros(k, dtype=np.int64)

        for j in range(k):
            idx_full = np.where(labels == j)[0]
            idx = idx_full if (sampled_per_cluster is None or sampled_per_cluster[j].size == 0) else \
            sampled_per_cluster[j]
            sizes[j] = idx.size
            if idx.size > 0:
                if weight_by_index is None:
                    w = weights[idx].reshape(-1, 1)
                else:
                    w = np.array([weight_by_index[int(ii)] for ii in idx], dtype=np.float64).reshape(-1, 1)
                # Eq.(7): weighted average
                centroids[j] = np.sum(w * X[idx], axis=0) / max(float(np.sum(w)), 1e-30)

        # Eq.(8): if any cluster is empty, reinitialize by farthest point from largest cluster centroid
        if np.any(sizes == 0):
            # largest cluster by size
            jmax = int(np.argmax(sizes))
            # choose farthest point from centroid of largest cluster (within that cluster)
            idx_max = np.where(labels == jmax)[0]
            if idx_max.size > 0:
                dists = cdist(X[idx_max], centroids[jmax][None, :], metric=self.metric).reshape(-1)
                farthest_local = idx_max[int(np.argmax(dists))]
                for j in range(k):
                    if sizes[j] == 0:
                        centroids[j] = X[farthest_local].astype(np.float64)
                        sizes[j] = 1  # mark as non-empty for this iteration

        # centroid movement (used in stopping criterion)
        movement = float(np.mean(np.linalg.norm(centroids - centroids_prev, axis=1)))
        return centroids, movement

    # -----------------------------
    # Main API: fit / predict (Appendix A.1)
    # -----------------------------
    def fit(self, X, iter_max=200):
        """
        训练 K-Sil 模型（Appendix A.1）

        备注：保持与原实验代码兼容：
            - 返回 centers（质心），并设置 self.centers / self.cluster_labels_
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        # (μ, L) ← Initial centroids and labels via one k-means iteration  (Appendix A.1)
        centroids = self._init_centroids(X)
        D0 = cdist(X, centroids, metric=self.metric)
        labels = np.argmin(D0, axis=1)

        # best solution tracking by silhouette objective S*  (Appendix A.1)
        S_star = -1.0
        best_centroids = centroids.copy()
        best_labels = labels.copy()

        for _ in range(int(iter_max)):
            # Objective-aligned sampling (§3.1, Appendix A.1)
            sampled_per = self._sample_indices_per_cluster(labels)
            sampled_idx = np.concatenate([s for s in sampled_per if s.size > 0], axis=0) if any(
                s.size > 0 for s in sampled_per) else np.arange(n_samples, dtype=np.int64)

            # Compute silhouette scores for sampled xi (Eq.(1) or Eq.(3)-(4))
            if self.use_approx_silhouette:
                sil_s = self._silhouette_approx(X[sampled_idx], labels[sampled_idx], centroids)
            else:
                sil_s = self._silhouette_exact(X[sampled_idx], labels[sampled_idx])

            # Assign weights for sampled xi based on scheme (Eq.(5)/(6))
            w_s = self._compute_weights(sil_s, labels[sampled_idx])
            weight_by_index = {int(ii): float(w) for ii, w in zip(sampled_idx, w_s)}

            # μ ← Update cluster centroids using sampled weighted averages (Eq.(7)), handle empty clusters (Eq.(8))
            # 说明：采样启用时，按 Appendix A.1：用 C_j^S 上的权重更新质心；若某簇采样为空，则退化为全簇均匀权重更新。
            weights_dummy = np.ones(n_samples, dtype=np.float64)  # 占位，不使用全量权重
            new_centroids, movement = self._update_centroids(X, labels, weights_dummy, centroids,
                                                             sampled_per_cluster=sampled_per,
                                                             weight_by_index=weight_by_index)

            # L ← Reassign each xi to nearest centroid
            D = cdist(X, new_centroids, metric=self.metric)
            new_labels = np.argmin(D, axis=1)

            # score ← Current silhouette aggregation objective on new Labels L  (Eq.(2), 采样对齐 §3.1)
            sampled_per_new = self._sample_indices_per_cluster(new_labels)
            sampled_idx_new = np.concatenate([s for s in sampled_per_new if s.size > 0], axis=0) if any(
                s.size > 0 for s in sampled_per_new) else np.arange(n_samples, dtype=np.int64)
            if self.use_approx_silhouette:
                sil_new = self._silhouette_approx(X[sampled_idx_new], new_labels[sampled_idx_new], new_centroids)
            else:
                sil_new = self._silhouette_exact(X[sampled_idx_new], new_labels[sampled_idx_new])

            score = self._aggregate_objective(sil_new, new_labels[sampled_idx_new])  # Eq.(2) on sampled points

            # retain best (μ*, L*) with max observed S* (Appendix A.1)
            if score > S_star:
                S_star = score
                best_centroids = new_centroids.copy()
                best_labels = new_labels.copy()

            centroids = new_centroids
            labels = new_labels

            # until (average centroid movement < τ) or max iterations reached
            if movement < self.tau:
                break

        self.centers = best_centroids
        self.cluster_labels_ = best_labels
        self.best_silhouette_objective_ = S_star  # 可用于对齐论文的 S*

        return self.centers

    def predict(self, X):
        X = np.asarray(X)
        D = cdist(X, self.centers, metric=self.metric)
        return np.argmin(D, axis=1)


# -------------------------------------------------------------------------
# 兼容包装：保持原实验流程不改动（仍实例化 K_Medoids）
# 说明：原文件中的 K_Medoids 实现是 k-medoids；此处为对比实验替换为 K-Sil。
# 为满足“准确命名”要求，真实实现类为 K_Sil_KMeans；K_Medoids 仅为旧接口适配。
# -------------------------------------------------------------------------
class K_Medoids(K_Sil_KMeans):
    pass


def accuracy_adjust(label1, label2):  # label1为真实标签，label2为预测标签
    new_label = np.zeros(len(label1))  # 该标签为预测标签调整后与真实标签对应的标签
    remove_repeat_label2 = list(set(label2))
    # print('去重后的预测标签:', remove_repeat_label2)
    for i in range(len(remove_repeat_label2)):
        label_location = [m for m, n in enumerate(label2) if n == remove_repeat_label2[i]]
        # print(remove_repeat_label2[i], label_location)
        location_for_label = []  # 预测标签里面标签remove_repeat_label2[i]对应的位置的真实标签
        for j in range(len(label_location)):
            location_for_label.append(label1[label_location[j]])
        maxlabel = max(location_for_label, key=location_for_label.count)  # 找出出现次数最多的元素
        for mm in range(len(label_location)):
            new_label[label_location[mm]] = maxlabel
    # print(new_label)
    return new_label

def unpickle(file):
    with open("./cifar/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# ==================== 主程序 ====================
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


    # ===== 欧氏距离 =====
    euc_max = np.max(Euc_count)
    euc_mean = np.mean(Euc_count)
    euc_sd = np.std(Euc_count, ddof=1)

    print('欧氏距离ksil聚类正确率：', Euc_count)
    print(f"欧氏距离 ksil：Max = {euc_max * 100:.1f}，Mean±SD = {euc_mean * 100:.1f}±{euc_sd * 100:.1f}")














