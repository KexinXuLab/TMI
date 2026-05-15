from hsi_indianpines_common import MethodSpec, import_paper_module, run_cli


def make_model(_seed: int):
    mod = import_paper_module("NSKmeans_mnist")
    return mod.NSK_Means(n_clusters=None, metric="euclidean", eps=0.1, verbose=False)


METHODS = [MethodSpec("NSK_MEANS_ED", "flat", make_model)]


if __name__ == "__main__":
    data_folder = './indianpines'
    raise SystemExit(run_cli(METHODS, data_folder, __file__))
