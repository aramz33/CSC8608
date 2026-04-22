import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
import torch
from torch_geometric.datasets import Planetoid


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    root = os.environ.get("PYG_DATA_ROOT", "/tmp/pyg_data")
    device = get_device()

    print(f"torch version  : {torch.__version__}")
    print(f"device         : {device}")
    if device.type == "cuda":
        print(f"gpu name       : {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"gpu name       : Apple MPS (M-series)")

    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    print(f"\n--- Cora dataset ---")
    print(f"num_nodes      : {data.num_nodes}")
    print(f"num_edges      : {data.num_edges}")
    print(f"num_features   : {dataset.num_node_features}")
    print(f"num_classes    : {dataset.num_classes}")
    print(f"train nodes    : {data.train_mask.sum().item()}")
    print(f"val nodes      : {data.val_mask.sum().item()}")
    print(f"test nodes     : {data.test_mask.sum().item()}")
    print(f"edge_index     : {data.edge_index.shape}")
    print(f"x              : {data.x.shape}")
    print(f"y              : {data.y.shape}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
