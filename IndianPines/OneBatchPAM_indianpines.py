from hsi_indianpines_common import MethodSpec, import_paper_module, run_cli


def make_model(_seed: int):
    mod = import_paper_module("OneBatchPAM_mnist")
    return mod.OneBatchPAMKMedoids(n_clusters=2, metric="cityblock", batch_size="auto", weight="debias")


METHODS = [MethodSpec("ONEBATCH_MD", "flat", make_model)]


if __name__ == "__main__":
    data_folder = './indianpines'
    raise SystemExit(run_cli(METHODS, data_folder, __file__))
