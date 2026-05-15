from hsi_indianpines_common import MethodSpec, import_paper_module, run_cli


def make_model(_seed: int):
    mod = import_paper_module("M_TMI_mnist_3_3")
    return mod.K_Means_ML(n_clusters=2)


METHODS = [MethodSpec("M_TMI", "M_TMI", make_model)]


if __name__ == "__main__":
    data_folder = './indianpines'
    raise SystemExit(run_cli(METHODS, data_folder, __file__))
