"""Build a state-action scoring RM: V(s, a) = P(score=1 | s, a).

For each of N_STATES random initial states, draw N_ACTIONS uniform
actions, simulate one shot under each, and record the triplet
(obs, action, score in {0,1}). Train a small MLP regressor on these
triples with BCE loss; this learned scorer is much more directly
useful as a SAC reward augmentation than V(s) because it gives a
direct credit signal for *which* action to take.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn

from billiards.inning_env import Billiards4BallInningEnv  # noqa: E402
from billiards.wrappers.random_start_env import RandomStartInningEnv  # noqa: E402


def make_env(max_shots: int = 50):
    base = Billiards4BallInningEnv(t_max=12.0, max_shots=max_shots)
    return RandomStartInningEnv(base)


def _snap(env):
    inner = env.unwrapped
    return (
        copy.deepcopy(inner._state),
        inner._shot_index,
        inner._cumulative_score,
        inner._cumulative_t,
        list(inner._shot_trajectories),
        list(inner._shot_offsets),
        list(inner._inning_log_records),
    )


def _restore(env, s):
    inner = env.unwrapped
    inner._state = copy.deepcopy(s[0])
    inner._shot_index = s[1]
    inner._cumulative_score = s[2]
    inner._cumulative_t = s[3]
    inner._shot_trajectories = list(s[4])
    inner._shot_offsets = list(s[5])
    inner._inning_log_records = list(s[6])


def collect_pairs(env, n_states: int, n_actions: int, seed: int = 0):
    low = env.action_space.low.astype(np.float64)
    high = env.action_space.high.astype(np.float64)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    total = n_states * n_actions

    obs_buf = np.zeros((total, obs_dim), dtype=np.float32)
    act_buf = np.zeros((total, act_dim), dtype=np.float32)
    score_buf = np.zeros(total, dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = 0
    pbar = tqdm(range(n_states), desc="collect", unit="state",
                mininterval=1.0, smoothing=0.05)
    for i in pbar:
        obs, _ = env.reset(seed=seed + i)
        snap = _snap(env)
        for _ in range(n_actions):
            _restore(env, snap)
            a = rng.uniform(low=low, high=high).astype(np.float32)
            try:
                _, r, _, _, _ = env.step(a)
            except Exception:
                r = 0.0
            obs_buf[idx] = obs
            act_buf[idx] = a
            score_buf[idx] = float(r > 0.5)
            idx += 1
        if (i + 1) % 100 == 0:
            pbar.set_postfix(pos_frac=f"{score_buf[:idx].mean():.4f}")
    pbar.close()
    return obs_buf, act_buf, score_buf


class VSaNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)  # logit


def train_vsa(obs, act, score, hidden: int = 256, epochs: int = 30,
              batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.1,
              device: str = "cpu") -> tuple[VSaNet, dict]:
    torch.manual_seed(0)
    n = obs.shape[0]
    perm = np.random.RandomState(0).permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    s_t = torch.tensor(score, dtype=torch.float32, device=device)

    net = VSaNet(obs_dim=obs.shape[1], act_dim=act.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train": [], "val_loss": [], "val_acc": [], "val_pos_pred": []}

    ep_bar = tqdm(range(epochs), desc="train RM", unit="ep", mininterval=1.0)
    for ep in ep_bar:
        net.train()
        perm_tr = np.random.permutation(len(tr_idx))
        total_loss = 0.0; nb = 0
        for b0 in range(0, len(tr_idx), batch_size):
            idx = tr_idx[perm_tr[b0:b0 + batch_size]]
            logit = net(obs_t[idx], act_t[idx])
            loss = loss_fn(logit, s_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()); nb += 1
        history["train"].append(total_loss / max(1, nb))
        net.eval()
        with torch.no_grad():
            vlogit = net(obs_t[val_idx], act_t[val_idx])
            vloss = float(loss_fn(vlogit, s_t[val_idx]).item())
            vprob = torch.sigmoid(vlogit)
            # AUC-like quick metric: mean predicted prob on positives vs negatives
            pos_mask = s_t[val_idx] > 0.5
            mp_pos = float(vprob[pos_mask].mean().item()) if pos_mask.any() else 0.0
            mp_neg = float(vprob[~pos_mask].mean().item())
            acc = float(((vprob > 0.5).float() == s_t[val_idx]).float().mean().item())
        history["val_loss"].append(vloss)
        history["val_acc"].append(acc)
        history["val_pos_pred"].append(mp_pos)
        ep_bar.set_postfix(train=f"{total_loss / max(1, nb):.4f}",
                           val=f"{vloss:.4f}",
                           mp_pos=f"{mp_pos:.3f}", mp_neg=f"{mp_neg:.3f}")
    ep_bar.close()
    return net, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_states", type=int, default=5000)
    parser.add_argument("--n_actions", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="experiments/rm_data_vsa")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    t0 = time.perf_counter()
    obs, act, score = collect_pairs(env, args.n_states, args.n_actions, seed=args.seed)
    print(f"[build_vsa] dataset {obs.shape[0]} pairs, "
          f"pos_rate={score.mean():.4f} in {time.perf_counter()-t0:.1f}s")

    # Save dataset (compact: float32)
    np.savez(out_dir / "vsa_dataset.npz", obs=obs, act=act, score=score)

    t1 = time.perf_counter()
    net, history = train_vsa(obs, act, score, hidden=args.hidden, epochs=args.epochs)
    print(f"[build_vsa] RM trained in {time.perf_counter()-t1:.1f}s")

    torch.save({
        "state_dict": net.state_dict(),
        "obs_dim": int(obs.shape[1]),
        "act_dim": int(act.shape[1]),
        "hidden": int(args.hidden),
        "history": history,
        "pos_rate": float(score.mean()),
    }, out_dir / "vsa_rm.pt")
    print(f"[build_vsa] saved -> {out_dir / 'vsa_rm.pt'}")
    print(f"[build_vsa] DONE wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
