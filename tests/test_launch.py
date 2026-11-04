import re

from launch import new_run_id, wandb_identity


def test_wandb_identity():
    assert wandb_identity("124M/EpiSelect", 3) == ("124M-EpiSelect", "124M-EpiSelect-seed3")
    assert wandb_identity(None, 3) == (None, None)


def test_new_run_id_format():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50 and all(re.fullmatch(r"[a-z0-9]{8}", i) for i in ids)


def _compose(experiment):
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="main", overrides=[f"+experiment={experiment}"])


def test_ablation_experiments_flip_one_flag():
    ado, ado_np = _compose("124M/ADO"), _compose("124M/ADO_noprior")
    assert ado.loader.use_prior is True and ado_np.loader.use_prior is False
    assert {k: v for k, v in ado.loader.items() if k != "use_prior"} == {
        k: v for k, v in ado_np.loader.items() if k != "use_prior"
    }
    epi, epi_d = _compose("124M/EpiSelect"), _compose("124M/EpiSelect_diag")
    assert epi.loader.diag_gamma is False and epi_d.loader.diag_gamma is True
    assert {k: v for k, v in epi.loader.items() if k != "diag_gamma"} == {
        k: v for k, v in epi_d.loader.items() if k != "diag_gamma"
    }
    assert epi.model == epi_d.model and ado.model == ado_np.model
