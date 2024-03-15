from .geometry_data import build_geometry_dataloader
from .partnet_data import build_partnet_dataloader


def build_dataloader(cfg):
    if cfg.data.dataset == "partnet":
        return build_partnet_dataloader(cfg)
    elif cfg.data.dataset == "geometry":
        return build_geometry_dataloader(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset} not supported")
