from hsi_indianpines_common import MethodSpec, import_paper_module, run_cli


def make_euclidean_model(_seed: int):
    mod = import_paper_module("k_medoids_mnist_EDMD")
    return mod.K_Medoids(n_clusters=2, metric="euclidean")


def make_cityblock_model(_seed: int):
    mod = import_paper_module("k_medoids_mnist_EDMD")
    return mod.K_Medoids(n_clusters=2, metric="cityblock")


METHODS = [
    MethodSpec("KMEDOIDS_ED", "flat", make_euclidean_model),
    MethodSpec("KMEDOIDS_MD", "flat", make_cityblock_model),
]


if __name__ == "__main__":
    data_folder = './indianpines'
    raise SystemExit(run_cli(METHODS, data_folder, __file__))
