"""
Improved BC training: cosine LR, label smoothing, early stopping
"""
from env.robust import set_seed

set_seed(42)

import numpy as np
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from models_v2 import ImprovedPolicyNet
from config import BC_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bc_v2(
    data_path="offline_dataset.npz",
    save_path="bc_v2.pt",
    use_reward_v2=False,
    **kwargs,
):
    cfg = {**BC_CONFIG, **kwargs}
    d = np.load(data_path)
    s = torch.FloatTensor(d["s"])
    a_raw = np.array(d["a"]).flatten()
    if np.issubdtype(a_raw.dtype, np.floating):
        action_space = d.get("action_space", np.array([0.0, 0.5, 1.0, 2.0]))
        a = torch.LongTensor(
            [np.argmin(np.abs(action_space - v)) for v in a_raw]
        )
    else:
        a = torch.LongTensor(a_raw.astype(np.int64))

    model = ImprovedPolicyNet(
        state_dim=4, action_dim=4, hidden=cfg["hidden_dim"]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    N = len(s)
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(cfg["epochs"]):
        idx = torch.randperm(N)
        total_loss = 0.0
        num_batches = 0
        for i in range(0, N, cfg["batch_size"]):
            batch_idx = idx[i : i + cfg["batch_size"]]
            s_b = s[batch_idx].to(DEVICE)
            a_b = a[batch_idx].to(DEVICE)
            logits = model(s_b)
            loss = criterion(logits, a_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        scheduler.step()
        avg_loss = total_loss / num_batches

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if no_improve >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | loss={avg_loss:.6f} | best={best_loss:.6f}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model


if __name__ == "__main__":
    if not os.path.exists("offline_dataset.npz"):
        from data.generate import generate_dataset_v2, save_dataset

        data = generate_dataset_v2(n_trajectories=500, use_reward_v2=True)
        save_dataset(data)
    train_bc_v2(use_reward_v2=True)
