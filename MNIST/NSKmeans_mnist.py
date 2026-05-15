
import numpy as np
import gzip
import os
from scipy.spatial.distance import cdist

def accuracy_adjust1(label1, label2):  # label1为真实标签，label2为预测标签
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

def load_data_gz(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
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
    """把聚类标签映射到真实标签空间（多数投票）"""
    new_label = np.zeros(len(label1), dtype=np.int64)
    uniq = list(set(label2.tolist() if hasattr(label2, "tolist") else list(label2)))
    for u in uniq:
        idx = np.where(label2 == u)[0]
        true_labels = label1[idx].tolist()
        maxlabel = max(true_labels, key=true_labels.count)
        new_label[idx] = maxlabel
    return new_label


# =========================================================
# Paper-inspired utilities (shared by the 5 methods)
#   Paper: Heidari et al., Pattern Recognition 155 (2024) 110639
# =========================================================

class _PaperBase:
    """
    这份实现尽量贴近论文描述的“核心机制”，并把关键环节打印出来方便你核对。
    论文里最关键的几件事：

    (A) 初始两中心（Eq.1-3）：
        - c1 = argmin(sum distance to all points)
        - c2 = argmax(sum distance to all points)  (作者用它把 outlier 先单独分出去)

    (B) 簇“边界/长度/空间/密度”（Eq.4-12）：
        - 不是简单 min/max！而是相对中心点的左右边界：
          x_{k,max+}：在每个维度上，所有 >= center 的点里最大值
          x_{k,max-}：在每个维度上，所有 <= center 的点里最小值
          length = x_{k,max+} - x_{k,max-}
          space = Π length
          density = Nk / space

    (C) 选新中心（Eq.13-14 / Sec.5）：
        - SMDK / CSK：从“最低密度簇”里选“簇内最远点”（用簇内距离和最大者）
        - NSK：从“最大 overlap space”里选一个点当下一中心

    (D) 停止条件：
        - SMDK（Eq.20）：连续两次选到同一个新中心，就停
        - CSK/NSK（Eq.21 + 文中实验描述）：看最大 overlap space 是否为 0，
          或 <= ε，或与上一轮变化很小（论文里还提到用 D·ε）

    (E) CSK 的 overlap 点重分配（Sec.4/5.2）：
        - overlap 区域内的点，分别尝试并入两边簇，哪个让“密度更大”，就归哪个簇

    ⚠️ 重要说明（必须说清楚）：
    - 论文没写“如果空间=0 怎么办”。但在 784 维（MNIST）里，很容易出现某些维度 length=0，
      space 直接变 0，会导致 density 爆炸/NaN。这里我只在“除法时”做数值保护：
      density = Nk / max(space, tiny)，并且会打印出“出现了 0-length 维度”的次数，方便你检查。
      这属于数值工程，不是算法思想改动。
    """

    def __init__(
        self,
        metric="euclidean",
        eps=0.1,                # 论文说 0~0.2 之间实验效果较好（并在实验里用到 D*eps）
        max_outer_iters=50,
        max_inner_iters=200,
        random_state=42,
        verbose=True,
        tiny=1e-12,             # 仅用于防止除零/NaN
        stop_delta_use_D=True,  # 是否按论文实验描述使用 D*eps 的“变化阈值”
    ):
        self.metric = metric
        self.eps = float(eps)
        self.max_outer_iters = int(max_outer_iters)
        self.max_inner_iters = int(max_inner_iters)
        self.random_state = int(random_state) if random_state is not None else None
        self.verbose = bool(verbose)
        self.tiny = float(tiny)
        self.stop_delta_use_D = bool(stop_delta_use_D)

    # ---------- distances ----------
    def _pairwise(self, X):
        return cdist(X, X, metric=self.metric)

    # ---------- Eq.(1)-(3): SMDK init of two centers ----------
    def _init_two_centers_smdk(self, D_xx):
        sum_dist = D_xx.sum(axis=1)
        c1 = int(np.argmin(sum_dist))
        c2 = int(np.argmax(sum_dist))
        if c2 == c1:
            c2 = int(np.argmax(D_xx[c1]))
        if self.verbose:
            print(f"[Init Eq(1-3)] c1(argmin sumDist)={c1}, c2(argmax sumDist)={c2}")
        return [c1, c2]

    # ---------- Eq.(4)-(12): bounds / length / space / density ----------
    def _cluster_bounds_by_center(self, X, labels, centers):
        """
        论文 Eq.(4)-(9) 的“左右边界”版本：
          low[k,d]  = x_{k,max-}  (<= center) 的最小值
          high[k,d] = x_{k,max+}  (>= center) 的最大值
        """
        k = int(labels.max()) + 1
        d = X.shape[1]
        low = np.zeros((k, d), dtype=np.float64)
        high = np.zeros((k, d), dtype=np.float64)
        counts = np.zeros(k, dtype=int)

        zero_len_dims = 0  # 统计出现 length==0 的维度数量（仅用于打印）

        for ci in range(k):
            idx = np.where(labels == ci)[0]
            counts[ci] = len(idx)
            if len(idx) == 0:
                # 空簇：用 center 自己做边界
                c = centers[ci].astype(np.float64)
                low[ci] = c
                high[ci] = c
                zero_len_dims += d
                continue

            pts = X[idx].astype(np.float64)
            c = centers[ci].astype(np.float64)

            # 对每个维度：右侧取 max(pts[pts>=c])，左侧取 min(pts[pts<=c])
            # 为了速度，这里向量化做：
            ge = pts >= c
            le = pts <= c

            # right_max: 如果某维度没有 >=c 的点，就用 c
            right = np.where(ge, pts, -np.inf).max(axis=0)
            right = np.where(np.isfinite(right), right, c)

            # left_min: 如果某维度没有 <=c 的点，就用 c
            left = np.where(le, pts, np.inf).min(axis=0)
            left = np.where(np.isfinite(left), left, c)

            low[ci] = left
            high[ci] = right

            zero_len_dims += int(np.sum((high[ci] - low[ci]) == 0))

        return low, high, counts, zero_len_dims

    def _cluster_space(self, low, high):
        lengths = (high - low).astype(np.float64)
        # 论文就是 length=0 就是 0；我们这里不改它，只在密度里防除零
        space = np.prod(lengths, axis=1)
        return space, lengths

    def _cluster_density(self, counts, space):
        return counts / np.maximum(space, self.tiny)

    # ---------- Eq.(15)-(19): overlap space ----------
    def _overlap_matrix(self, low, high):
        """
        overlap 的“体积”：对每对簇 (i,j)，
          inter_len_d = max(0, min(high_i, high_j) - max(low_i, low_j))
          S_ij = Π inter_len_d
        """
        k, d = low.shape
        S = np.zeros((k, k), dtype=np.float64)
        boxes = [[None] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                lo = np.maximum(low[i], low[j])
                hi = np.minimum(high[i], high[j])
                lengths = (hi - lo).astype(np.float64)
                if np.all(lengths > 0):
                    vol = float(np.prod(lengths))
                    S[i, j] = vol
                    S[j, i] = vol
                    boxes[i][j] = (lo, hi)
                    boxes[j][i] = (lo, hi)
        return S, boxes

    def _points_in_box(self, X, lo, hi, candidate_idx=None):
        if candidate_idx is None:
            candidate_idx = np.arange(X.shape[0])
        pts = X[candidate_idx]
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        return candidate_idx[np.where(mask)[0]]

    # ---------- Eq.(13)-(14): next center from lowest-density cluster ----------
    def _next_center_from_low_density(self, D_xx, labels, densities):
        low_k = int(np.argmin(densities))
        idx = np.where(labels == low_k)[0]
        if len(idx) == 0:
            return int(np.argmax(D_xx.sum(axis=1)))
        sub = D_xx[np.ix_(idx, idx)]
        sums = sub.sum(axis=1)
        return int(idx[np.argmax(sums)])

    # ---------- NSK: next center from max overlap space ----------
    def _next_center_from_max_overlap(self, X, D_xx, centers_idx, S, boxes, labels, densities):
        i, j = np.unravel_index(np.argmax(S), S.shape)
        lo_hi = boxes[i][j]
        if lo_hi is None:
            if self.verbose:
                print("[NSK] No overlap box found (unexpected). Fallback to low-density rule.")
            return self._next_center_from_low_density(D_xx, labels, densities)

        lo, hi = lo_hi
        idx_i = np.where(labels == i)[0]
        idx_j = np.where(labels == j)[0]
        cand = np.unique(np.concatenate([idx_i, idx_j]))
        pts = self._points_in_box(X, lo, hi, candidate_idx=cand)
        if len(pts) == 0:
            if self.verbose:
                print("[NSK] Max-overlap region contains 0 points. Fallback to low-density rule.")
            return self._next_center_from_low_density(D_xx, labels, densities)

        # 论文说“选 overlap 里的一个点作为下一中心”，未规定细节；这里选“离已有中心最远”的，确定性强
        dist_to_any = D_xx[np.ix_(pts, centers_idx)].min(axis=1)
        return int(pts[np.argmax(dist_to_any)])

    # ---------- CSK: reassign overlap points by density comparison ----------
    def _csk_reassign_overlap_points(self, X, labels, low, high, counts):
        """
        贴论文 Sec.4/5.2：
        overlap 区域点 p：
          - 假设把 p 放进簇 i（更新 i 的边界/space/density）
          - 假设把 p 放进簇 j
          - 谁密度更高，p 就归谁
        为了加速，只在“i簇∪j簇”里找 overlap 点（不是全体扫描）。
        """
        k = int(labels.max()) + 1
        S, boxes = self._overlap_matrix(low, high)

        moved = 0
        checked = 0

        for i in range(k):
            for j in range(i + 1, k):
                if S[i, j] <= 0:
                    continue
                lo, hi = boxes[i][j]
                idx_i = np.where(labels == i)[0]
                idx_j = np.where(labels == j)[0]
                cand = np.unique(np.concatenate([idx_i, idx_j]))
                pts = self._points_in_box(X, lo, hi, candidate_idx=cand)
                if len(pts) == 0:
                    continue

                for p in pts:
                    checked += 1
                    xp = X[p].astype(np.float64)

                    # i 假设
                    low_i = np.minimum(low[i], xp)
                    high_i = np.maximum(high[i], xp)
                    len_i = high_i - low_i
                    space_i = float(np.prod(len_i))
                    n_i = counts[i] + (0 if labels[p] == i else 1)
                    rho_i = n_i / max(space_i, self.tiny)

                    # j 假设
                    low_j = np.minimum(low[j], xp)
                    high_j = np.maximum(high[j], xp)
                    len_j = high_j - low_j
                    space_j = float(np.prod(len_j))
                    n_j = counts[j] + (0 if labels[p] == j else 1)
                    rho_j = n_j / max(space_j, self.tiny)

                    new_lab = i if rho_i >= rho_j else j
                    if new_lab != labels[p]:
                        moved += 1
                        labels[p] = new_lab

        if self.verbose:
            print(f"[CSK] overlap reassignment: checked={checked}, moved={moved}")
        return labels


# =========================================================
# 1) SMDK-means
# 2) SMDK-medoids
# 3) CSK-means
# 4) NSK-means
# 5) NSK-medoids
# =========================================================

class SMDK_Medoids(_PaperBase):
    """SMDK-medoids: SMDK 选中心 + K-medoids 内循环；停止条件按 Eq.(20)。"""

    def __init__(self, n_clusters=None, metric="euclidean", **kwargs):
        super().__init__(metric=metric, **kwargs)
        self.n_clusters = None if n_clusters is None else int(n_clusters)

    def _kmedoids_fixedK(self, D_xx, medoid_indices):
        X = self._X
        medoid_indices = np.array(medoid_indices, dtype=int)

        for it in range(self.max_inner_iters):
            prev = medoid_indices.copy()
            D_to = D_xx[:, medoid_indices]
            labels = np.argmin(D_to, axis=1)

            new_meds = medoid_indices.copy()
            for k in range(len(medoid_indices)):
                idx = np.where(labels == k)[0]
                if len(idx) == 0:
                    dist_to_any = D_xx[:, new_meds].min(axis=1)
                    dist_to_any[new_meds] = -np.inf
                    new_meds[k] = int(np.argmax(dist_to_any))
                    continue
                if len(idx) == 1:
                    new_meds[k] = int(idx[0])
                    continue
                sub = D_xx[np.ix_(idx, idx)]
                new_meds[k] = int(idx[np.argmin(sub.sum(axis=1))])

            medoid_indices = new_meds
            if np.array_equal(prev, medoid_indices):
                break

        centers = X[medoid_indices].astype(np.float64).copy()
        return medoid_indices, centers, labels

    def _stop_csk_nsk(self, smax, prev_smax, d):
        # Eq.(21) + 文中实验描述：也考虑 D*eps 的变化阈值
        if smax == 0.0:
            return True, "ideal Smax=0"
        if smax <= self.eps:
            return True, f"Smax<=eps ({smax:.3g}<={self.eps:.3g})"
        if prev_smax is None:
            return False, ""
        if np.isclose(smax, prev_smax):
            return True, "Smax equals previous"
        if self.stop_delta_use_D:
            if abs(smax - prev_smax) <= d * self.eps:
                return True, f"|ΔSmax|<=D*eps ({abs(smax-prev_smax):.3g}<={d*self.eps:.3g})"
            if smax > prev_smax:
                return True, "Smax increased"
        return False, ""

    def fit(self, X):
        X = np.asarray(X)
        self._X = X
        D_xx = self._pairwise(X)
        d = X.shape[1]

        # fixed K（与你现有 main 保持兼容）
        if self.n_clusters is not None:
            meds = self._init_two_centers_smdk(D_xx)
            while len(meds) < self.n_clusters:
                med_idx, ctrs, labels = self._kmedoids_fixedK(D_xx, meds)
                low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, ctrs)
                space, _ = self._cluster_space(low, high)
                dens = self._cluster_density(counts, space)
                nxt = self._next_center_from_low_density(D_xx, labels, dens)
                if nxt in meds:
                    dist_to_any = D_xx[:, meds].min(axis=1)
                    dist_to_any[meds] = -np.inf
                    nxt = int(np.argmax(dist_to_any))
                meds.append(nxt)

            med_idx, ctrs, labels = self._kmedoids_fixedK(D_xx, meds)
            self.center_indices = med_idx
            self.centers = ctrs
            self.cluster_labels_ = labels
            return ctrs

        # auto-K（SMDK stop: Eq.20）
        meds = self._init_two_centers_smdk(D_xx)
        prev_added = None

        for outer in range(self.max_outer_iters):
            med_idx, ctrs, labels = self._kmedoids_fixedK(D_xx, meds)
            k = len(meds)

            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, ctrs)
            space, _ = self._cluster_space(low, high)
            dens = self._cluster_density(counts, space)

            if self.verbose:
                print(f"[SMDK-medoids outer={outer}] K={k} zeroLenDims={z0} dens(min/med/max)={dens.min():.3g}/{np.median(dens):.3g}/{dens.max():.3g}")

            nxt = self._next_center_from_low_density(D_xx, labels, dens)

            # Eq.(20)
            if prev_added is not None and nxt == prev_added:
                if self.verbose:
                    print(f"[SMDK-medoids stop Eq(20)] next center repeats: {nxt}")
                self.center_indices = med_idx
                self.centers = ctrs
                self.cluster_labels_ = labels
                return ctrs

            prev_added = nxt

            if nxt in meds:
                if self.verbose:
                    print("[SMDK-medoids stop] next center already in set -> stop")
                self.center_indices = med_idx
                self.centers = ctrs
                self.cluster_labels_ = labels
                return ctrs

            meds.append(nxt)

        self.center_indices = med_idx
        self.centers = ctrs
        self.cluster_labels_ = labels
        return ctrs

    def predict(self, X):
        X = np.asarray(X)
        D = cdist(X, self.centers, metric=self.metric)
        return np.argmin(D, axis=1)


class NSK_Medoids(SMDK_Medoids):
    """NSK-medoids：换“下一中心选择”为最大 overlap；停止条件同 CSK/NSK（Eq.21）。"""

    def fit(self, X):
        X = np.asarray(X)
        self._X = X
        D_xx = self._pairwise(X)
        d = X.shape[1]

        # fixed K：论文主要讨论 auto-K；这里保持和原实验兼容（固定K就不走 NSK 的加中心规则）
        if self.n_clusters is not None:
            return super().fit(X)

        meds = self._init_two_centers_smdk(D_xx)
        prev_smax = None

        for outer in range(self.max_outer_iters):
            med_idx, ctrs, labels = self._kmedoids_fixedK(D_xx, meds)
            k = len(meds)

            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, ctrs)
            space, _ = self._cluster_space(low, high)
            dens = self._cluster_density(counts, space)
            S, boxes = self._overlap_matrix(low, high)
            smax = float(S.max()) if k > 1 else 0.0

            if self.verbose:
                print(f"[NSK-medoids outer={outer}] K={k} Smax={smax:.6g} zeroLenDims={z0}")

            stop, reason = self._stop_csk_nsk(smax, prev_smax, d)
            if stop:
                if self.verbose:
                    print(f"[NSK-medoids stop] {reason}")
                self.center_indices = med_idx
                self.centers = ctrs
                self.cluster_labels_ = labels
                return ctrs

            prev_smax = smax
            nxt = self._next_center_from_max_overlap(X, D_xx, meds, S, boxes, labels, dens)

            if self.verbose:
                print(f"[NSK-medoids] next center(from max overlap) = {nxt}")

            if nxt in meds:
                if self.verbose:
                    print("[NSK-medoids stop] next center already exists -> stop")
                self.center_indices = med_idx
                self.centers = ctrs
                self.cluster_labels_ = labels
                return ctrs

            meds.append(nxt)

        self.center_indices = med_idx
        self.centers = ctrs
        self.cluster_labels_ = labels
        return ctrs


class SMDK_Means(_PaperBase):
    """SMDK-means: SMDK 选中心 + K-means 内循环；停止条件按 Eq.(20)。"""

    def __init__(self, n_clusters=None, metric="euclidean", **kwargs):
        super().__init__(metric=metric, **kwargs)
        self.n_clusters = None if n_clusters is None else int(n_clusters)

    def _kmeans_fixedK_by_centers(self, X, centers):
        centers = np.asarray(centers, dtype=np.float64).copy()
        for it in range(self.max_inner_iters):
            prev = centers.copy()
            D = cdist(X, centers, metric=self.metric)
            labels = np.argmin(D, axis=1)

            new_centers = centers.copy()
            for k in range(centers.shape[0]):
                idx = np.where(labels == k)[0]
                if len(idx) == 0:
                    dist_to_any = D.min(axis=1)
                    new_centers[k] = X[int(np.argmax(dist_to_any))].astype(np.float64)
                    continue
                if self.metric == "cityblock":
                    new_centers[k] = np.median(X[idx].astype(np.float64), axis=0)
                else:
                    new_centers[k] = X[idx].astype(np.float64).mean(axis=0)

            centers = new_centers
            if np.allclose(prev, centers, rtol=0, atol=1e-9):
                break
        return centers, labels

    def _kmeans_fixedK_by_indices(self, X, D_xx, center_indices):
        centers = X[np.array(center_indices, dtype=int)].astype(np.float64).copy()
        return self._kmeans_fixedK_by_centers(X, centers)

    def fit(self, X):
        X = np.asarray(X)
        self._X = X
        D_xx = self._pairwise(X)

        # fixed K
        if self.n_clusters is not None:
            centers_idx = self._init_two_centers_smdk(D_xx)
            while len(centers_idx) < self.n_clusters:
                centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
                low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, centers)
                space, _ = self._cluster_space(low, high)
                dens = self._cluster_density(counts, space)
                nxt = self._next_center_from_low_density(D_xx, labels, dens)
                if nxt in centers_idx:
                    dist_to_any = cdist(X, centers, metric=self.metric).min(axis=1)
                    dist_to_any[centers_idx] = -np.inf
                    nxt = int(np.argmax(dist_to_any))
                centers_idx.append(nxt)

            centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
            self.centers = centers
            self.cluster_labels_ = labels
            return centers

        # auto-K with Eq.(20)
        centers_idx = self._init_two_centers_smdk(D_xx)
        prev_added = None

        for outer in range(self.max_outer_iters):
            centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
            k = len(centers_idx)
            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, centers)
            space, _ = self._cluster_space(low, high)
            dens = self._cluster_density(counts, space)

            if self.verbose:
                print(f"[SMDK-means outer={outer}] K={k} zeroLenDims={z0} dens(min/med/max)={dens.min():.3g}/{np.median(dens):.3g}/{dens.max():.3g}")

            nxt = self._next_center_from_low_density(D_xx, labels, dens)

            if prev_added is not None and nxt == prev_added:
                if self.verbose:
                    print(f"[SMDK-means stop Eq(20)] next center repeats: {nxt}")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            prev_added = nxt

            if nxt in centers_idx:
                if self.verbose:
                    print("[SMDK-means stop] next center already in set -> stop")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            centers_idx.append(nxt)

        self.centers = centers
        self.cluster_labels_ = labels
        return centers

    def predict(self, X):
        X = np.asarray(X)
        D = cdist(X, self.centers, metric=self.metric)
        return np.argmin(D, axis=1)


class CSK_Means(SMDK_Means):
    """CSK-means = SMDK-means + overlap 点重分配；停止条件按 Eq.(21)（以及文中 D*eps 版本）"""

    def _stop_csk_nsk(self, smax, prev_smax, d):
        if smax == 0.0:
            return True, "ideal Smax=0"
        if smax <= self.eps:
            return True, f"Smax<=eps ({smax:.3g}<={self.eps:.3g})"
        if prev_smax is None:
            return False, ""
        if np.isclose(smax, prev_smax):
            return True, "Smax equals previous"
        if self.stop_delta_use_D:
            if abs(smax - prev_smax) <= d * self.eps:
                return True, f"|ΔSmax|<=D*eps ({abs(smax-prev_smax):.3g}<={d*self.eps:.3g})"
            if smax > prev_smax:
                return True, "Smax increased"
        return False, ""

    def fit(self, X):
        X = np.asarray(X)
        self._X = X
        D_xx = self._pairwise(X)
        d = X.shape[1]

        # fixed K：kmeans 收敛 -> overlap 重分配 -> 重新计算中心并 refine
        if self.n_clusters is not None:
            centers_idx = self._init_two_centers_smdk(D_xx)
            while len(centers_idx) < self.n_clusters:
                centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
                low, high, counts, _ = self._cluster_bounds_by_center(X, labels, centers)
                space, _ = self._cluster_space(low, high)
                dens = self._cluster_density(counts, space)
                nxt = self._next_center_from_low_density(D_xx, labels, dens)
                if nxt in centers_idx:
                    dist_to_any = cdist(X, centers, metric=self.metric).min(axis=1)
                    dist_to_any[centers_idx] = -np.inf
                    nxt = int(np.argmax(dist_to_any))
                centers_idx.append(nxt)

            centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
            low, high, counts, _ = self._cluster_bounds_by_center(X, labels, centers)
            labels2 = self._csk_reassign_overlap_points(X, labels.copy(), low, high, counts)

            # 重新算中心，再 refine 一次
            init_centers = []
            for k in range(self.n_clusters):
                idx = np.where(labels2 == k)[0]
                if len(idx) == 0:
                    init_centers.append(centers[k])
                else:
                    if self.metric == "cityblock":
                        init_centers.append(np.median(X[idx].astype(np.float64), axis=0))
                    else:
                        init_centers.append(X[idx].astype(np.float64).mean(axis=0))
            centers3, labels3 = self._kmeans_fixedK_by_centers(X, np.vstack(init_centers))

            self.centers = centers3
            self.cluster_labels_ = labels3
            return centers3

        # auto-K：每轮 kmeans 收敛 -> overlap 重分配 -> 计算 Smax -> 停止/加中心
        centers_idx = self._init_two_centers_smdk(D_xx)
        prev_smax = None

        for outer in range(self.max_outer_iters):
            centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
            k = len(centers_idx)

            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, centers)
            labels = self._csk_reassign_overlap_points(X, labels.copy(), low, high, counts)

            # 重算 bounds/density/overlap
            centers, labels = self._kmeans_fixedK_by_centers(X, centers)  # 再 refine，防止 labels 影响中心
            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, centers)
            space, _ = self._cluster_space(low, high)
            dens = self._cluster_density(counts, space)
            S, _ = self._overlap_matrix(low, high)
            smax = float(S.max()) if k > 1 else 0.0

            if self.verbose:
                print(f"[CSK-means outer={outer}] K={k} Smax={smax:.6g} zeroLenDims={z0}")

            stop, reason = self._stop_csk_nsk(smax, prev_smax, d)
            if stop:
                if self.verbose:
                    print(f"[CSK-means stop] {reason}")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            prev_smax = smax
            nxt = self._next_center_from_low_density(D_xx, labels, dens)
            if self.verbose:
                print(f"[CSK-means] next center(from low density) = {nxt}")

            if nxt in centers_idx:
                if self.verbose:
                    print("[CSK-means stop] next center already exists -> stop")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            centers_idx.append(nxt)

        self.centers = centers
        self.cluster_labels_ = labels
        return centers


class NSK_Means(SMDK_Means):
    """NSK-means = SMDK-means + max-overlap 选下一中心；停止条件同 CSK/NSK（Eq.21）。"""

    def _stop_csk_nsk(self, smax, prev_smax, d):
        if smax == 0.0:
            return True, "ideal Smax=0"
        if smax <= self.eps:
            return True, f"Smax<=eps ({smax:.3g}<={self.eps:.3g})"
        if prev_smax is None:
            return False, ""
        if np.isclose(smax, prev_smax):
            return True, "Smax equals previous"
        if self.stop_delta_use_D:
            if abs(smax - prev_smax) <= d * self.eps:
                return True, f"|ΔSmax|<=D*eps ({abs(smax-prev_smax):.3g}<={d*self.eps:.3g})"
            if smax > prev_smax:
                return True, "Smax increased"
        return False, ""

    def fit(self, X):
        X = np.asarray(X)
        self._X = X
        D_xx = self._pairwise(X)
        d = X.shape[1]

        if self.n_clusters is not None:
            # 固定 K 时，为了对齐你现有实验，继续用 SMDK 的固定 K 策略
            return super().fit(X)

        centers_idx = self._init_two_centers_smdk(D_xx)
        prev_smax = None

        for outer in range(self.max_outer_iters):
            centers, labels = self._kmeans_fixedK_by_indices(X, D_xx, centers_idx)
            k = len(centers_idx)

            low, high, counts, z0 = self._cluster_bounds_by_center(X, labels, centers)
            space, _ = self._cluster_space(low, high)
            dens = self._cluster_density(counts, space)
            S, boxes = self._overlap_matrix(low, high)
            smax = float(S.max()) if k > 1 else 0.0

            if self.verbose:
                print(f"[NSK-means outer={outer}] K={k} Smax={smax:.6g} zeroLenDims={z0}")

            stop, reason = self._stop_csk_nsk(smax, prev_smax, d)
            if stop:
                if self.verbose:
                    print(f"[NSK-means stop] {reason}")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            prev_smax = smax
            nxt = self._next_center_from_max_overlap(X, D_xx, centers_idx, S, boxes, labels, dens)

            if self.verbose:
                print(f"[NSK-means] next center(from max overlap) = {nxt}")

            if nxt in centers_idx:
                if self.verbose:
                    print("[NSK-means stop] next center already exists -> stop")
                self.centers = centers
                self.cluster_labels_ = labels
                return centers

            centers_idx.append(nxt)

        self.centers = centers
        self.cluster_labels_ = labels
        return centers


# =========================================================
# Main (kept compatible with your current experiment code)
# =========================================================
if __name__ == '__main__':
    np.random.seed(42)

    data_folder = './MNIST_data'
    x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)

    photo_count = 1000
    a1 = 4900
    a2 = 4900

    loca_3 = np.where(y_train_gz == 3)
    loca_5 = np.where(y_train_gz == 5)

    label_3 = loca_3[0][a1:a1 + int(photo_count / 2)]
    label_5 = loca_5[0][a2:a2 + int(photo_count / 2)]

    label_all = []
    for i in range(len(label_3)):
        label_all.append(label_3[i])
        label_all.append(label_5[i])

    X = x_train_gz[label_all]
    y = y_train_gz[label_all]

    print(f"混合数据集形状: X={X.shape}, y={y.shape}")
    print(f"数字3的数量: {np.sum(y == 3)}, 数字5的数量: {np.sum(y == 5)}")

    recycle = 100

    # =========================
    # 选择要跑的论文方法（五选一）
    # =========================
    # 1) SMDK-means:
    # clusterer = SMDK_Means(n_clusters=None, metric='euclidean', eps=0.1, verbose=True)
    # 2) SMDK-medoids:
    # clusterer = SMDK_Medoids(n_clusters=None, metric='euclidean', eps=0.1, verbose=True)
    # 3) CSK-means:
    #clusterer = CSK_Means(n_clusters=None, metric='euclidean', eps=0.1, verbose=True)
    # 4) NSK-means:
    clusterer = NSK_Means(n_clusters=None, metric='euclidean', eps=0.1, verbose=True)
    # 5) NSK-medoids:
    # clusterer = NSK_Medoids(n_clusters=None, metric='euclidean', eps=0.1, verbose=True)

    acc_list = []
    for t in range(recycle):
        if clusterer.verbose:
            print(f"\n========== Run {t+1}/{recycle} ==========")
        clusterer.fit(X)
        labels = clusterer.predict(X) if hasattr(clusterer, "predict") else clusterer.cluster_labels_
        adjusted = accuracy_adjust1(y, labels)
        acc = float(np.sum(adjusted == y) / len(y))
        print("调整后准确率:", acc)  # 应为 1.0
        acc_list.append(acc)

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

    acc_list = np.array(acc_list, dtype=np.float64)
    print("\nAcc list:", acc_list)
    print("Max / Mean / SD:", acc_list.max(), acc_list.mean(), acc_list.std(ddof=1))
    print(f"Mean±SD: {acc_list.mean()*100:.1f}±{acc_list.std(ddof=1)*100:.1f}")
