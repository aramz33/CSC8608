import os
import ssl
import certifi
from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


@dataclass
class CoraData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int
    pyg_data: Data


os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def load_cora(root: str | None = None) -> CoraData:
    root = root or os.environ.get("PYG_DATA_ROOT", "/tmp/pyg_data")
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    return CoraData(
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        pyg_data=data,
    )
