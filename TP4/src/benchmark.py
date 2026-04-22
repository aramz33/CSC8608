import argparse
import os
import sys
import time
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import get_device


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=100)
    return p.parse_args()


def build_model(model_type: str, config_path: str, cora) -> torch.nn.Module:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if model_type == "mlp":
        mcfg = cfg["mlp"]
        return MLP(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])
    elif model_type == "gcn":
        mcfg = cfg["gcn"]
        return GCN(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])
    else:
        mcfg = cfg["sage"]
        return GraphSAGE(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])


@torch.no_grad()
def forward_once(model, x, edge_index, model_type):
    if model_type == "mlp":
        return model(x)
    return model(x, edge_index)


def main():
    args = parse_args()
    device = get_device()
    cora = load_cora()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_type = ckpt["model"]
    config_path = ckpt["config_path"]

    model = build_model(model_type, config_path, cora)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    x = cora.x.to(device)
    edge_index = cora.edge_index.to(device)
    num_nodes = cora.x.shape[0]

    warmup = args.warmup
    runs = args.runs

    print(f"model          : {model_type}")
    print(f"device         : {device}")
    print(f"num_nodes      : {num_nodes}")
    print(f"warmup iters   : {warmup}")
    print(f"timed iters    : {runs}")

    for _ in range(warmup):
        forward_once(model, x, edge_index, model_type)
    sync_device(device)

    times = []
    for _ in range(runs):
        sync_device(device)
        t0 = time.perf_counter()
        forward_once(model, x, edge_index, model_type)
        sync_device(device)
        times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000
    per_node_us = avg_ms * 1000 / num_nodes
    print(f"\navg_forward_ms : {avg_ms:.3f} ms")
    print(f"ms_per_node_approx: {per_node_us:.4f} µs/node")


if __name__ == "__main__":
    main()
