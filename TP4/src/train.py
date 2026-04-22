import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))

from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import get_device, set_seed, compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mlp", "gcn", "sage"])
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch_fullbatch(model, data, optimizer, criterion, device, model_type):
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device) if model_type != "mlp" else None
    mask = data.train_mask.to(device)

    logits = model(x) if model_type == "mlp" else model(x, edge_index)
    loss = criterion(logits[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch_sage(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        loss = criterion(seed_logits, seed_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, data, device, model_type):
    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device) if model_type != "mlp" else None
    logits = model(x) if model_type == "mlp" else model(x, edge_index)
    logits_cpu = logits.cpu()
    y_cpu = y.cpu()
    train_m = compute_metrics(logits_cpu, y_cpu, data.train_mask)
    val_m = compute_metrics(logits_cpu, y_cpu, data.val_mask)
    test_m = compute_metrics(logits_cpu, y_cpu, data.test_mask)
    return train_m, val_m, test_m


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    set_seed(cfg["seed"])

    print(f"model          : {args.model}")
    print(f"config         : {args.config}")
    print(f"device         : {device}")
    print(f"epochs         : {cfg['epochs']}")

    cora = load_cora()
    criterion = nn.CrossEntropyLoss()

    if args.model == "mlp":
        mcfg = cfg["mlp"]
        model = MLP(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])
    elif args.model == "gcn":
        mcfg = cfg["gcn"]
        model = GCN(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])
    else:
        mcfg = cfg["sage"]
        model = GraphSAGE(cora.num_features, mcfg["hidden_dim"], cora.num_classes, mcfg["dropout"])

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_loader = None
    if args.model == "sage":
        scfg = cfg["sampling"]
        train_loader = NeighborLoader(
            cora.pyg_data,
            input_nodes=cora.train_mask,
            num_neighbors=[scfg["num_neighbors_l1"], scfg["num_neighbors_l2"]],
            batch_size=scfg["batch_size"],
            shuffle=True,
        )

    total_start = time.perf_counter()
    loop_start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        if args.model == "sage":
            loss = train_epoch_sage(model, train_loader, optimizer, criterion, device)
        else:
            loss = train_epoch_fullbatch(model, cora, optimizer, criterion, device, args.model)
        epoch_time = time.perf_counter() - t0

        if epoch % 50 == 0 or epoch == 1:
            train_m, val_m, test_m = evaluate(model, cora, device, args.model)
            print(
                f"epoch {epoch:4d} | loss {loss:.4f} | "
                f"train_acc {train_m['acc']:.4f} | val_acc {val_m['acc']:.4f} | "
                f"epoch_time {epoch_time*1000:.1f}ms"
            )

    loop_time = time.perf_counter() - loop_start
    sync_device(device)
    total_time = time.perf_counter() - total_start

    train_m, val_m, test_m = evaluate(model, cora, device, args.model)
    print(f"\n--- Final metrics ---")
    print(f"test_acc       : {test_m['acc']:.4f}")
    print(f"test_macro_f1  : {test_m['f1']:.4f}")
    print(f"val_acc        : {val_m['acc']:.4f}")
    print(f"train_loop_time: {loop_time:.2f}s")
    print(f"total_train_time_s: {total_time:.2f}s")

    runs_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    ckpt_path = os.path.join(runs_dir, f"{args.model}.pt")
    torch.save({"model": args.model, "config_path": args.config, "state_dict": model.state_dict()}, ckpt_path)
    print(f"checkpoint     : {ckpt_path}")


if __name__ == "__main__":
    main()
