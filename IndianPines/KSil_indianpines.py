from hsi_indianpines_common import MethodSpec, import_paper_module, run_cli


def make_model(seed: int):
    mod = import_paper_module("KSil_mnist")
    return mod.K_Medoids(n_clusters=2, metric="euclidean", random_state=seed)


METHODS = [MethodSpec("K_SIL_ED", "flat", make_model)]


if __name__ == "__main__":
    data_folder = './indianpines'
    raise SystemExit(run_cli(METHODS, data_folder, __file__))
