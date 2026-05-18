"""
Safe CQL: Lagrangian Constrained Offline RL
CMDP: max E[sum r] s.t. E[sum c] <= epsilon

论文算法对应: src/algos/safe_cql.py 中的 update() 与 Lagrangian Loss
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algos.base import BaseAlgo
from src.models import Actor, Critic, SafetyCritic


class SafeCQL(BaseAlgo):
    """
    Lagrangian Safe CQL（双重保守）.
    update() 包含: Reward Q (CQL 压低 OOD Q_R)、Cost Q (CQL 推高 OOD Q_C + TD 目标用 max 悲观 bootstrap)、
    Lagrange、Actor 四步。
    """

    def __init__(
        self,
        state_dim=4,
        n_actions=4,
        hidden=64,
        actor_lr=1e-4,
        critic_lr=3e-4,
        lagrangian_lr=1e-3,
        gamma=0.99,
        alpha_cql=5.0,
        cost_limit=0.1,
        target_update_interval=2000,
        n_critics=2,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.alpha_cql = alpha_cql
        self.cost_limit = cost_limit
        self.target_update_interval = target_update_interval
        self.n_critics = n_critics
        self.n_actions = n_actions
        self.action_values = [0.0, 0.5, 1.0, 2.0]
        self._step = 0

        self.actor = Actor(state_dim, n_actions, hidden).to(self.device)
        self.q_r_nets = [Critic(state_dim, n_actions, hidden).to(
            self.device) for _ in range(n_critics)]
        self.q_r_targets = [Critic(state_dim, n_actions, hidden).to(
            self.device) for _ in range(n_critics)]
        self.q_c_nets = [SafetyCritic(state_dim, n_actions, hidden).to(
            self.device) for _ in range(n_critics)]
        self.q_c_targets = [SafetyCritic(state_dim, n_actions, hidden).to(
            self.device) for _ in range(n_critics)]
        for i in range(n_critics):
            self.q_r_targets[i].load_state_dict(self.q_r_nets[i].state_dict())
            self.q_c_targets[i].load_state_dict(self.q_c_nets[i].state_dict())

        self.log_lambda = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.tau = 0.005  # soft target update
        self.opt_qr = torch.optim.Adam(
            [p for net in self.q_r_nets for p in net.parameters()], lr=critic_lr)
        self.opt_qc = torch.optim.Adam(
            [p for net in self.q_c_nets for p in net.parameters()], lr=critic_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_lambda = torch.optim.Adam([self.log_lambda], lr=lagrangian_lr)

    def update(self, ss, aa, rr, cc, ss_next, term_b, return_metrics: bool = False):
        """
        单步更新。对应论文中的 Lagrangian Loss:
        L = L_CQL(Q_R) + L_TD(Q_C) + L_CQL-cost(Q_C) + L_λ(λ) + L_π(π)

        return_metrics=False 时不做 .item()/同步，训练环里只在 log 步为 True。
        """
        aa_onehot = F.one_hot(aa, self.n_actions).float()
        k = self.n_critics
        aa_idx_b = aa.unsqueeze(0).unsqueeze(-1).expand(k, -1, 1)

        # --- 1. Reward Q (CQL conservative)；本段内只前向 q_r_nets(ss) 一次 ---
        q_r_all = torch.stack([net(ss) for net in self.q_r_nets], dim=0)
        q_r = q_r_all.gather(2, aa_idx_b).squeeze(-1).mean(0)
        with torch.no_grad():
            next_a = self.actor(ss_next).argmax(-1)
            tgt_r = torch.stack([t(ss_next) for t in self.q_r_targets], dim=0)
            na_r = next_a.unsqueeze(0).unsqueeze(-1).expand(k, -1, 1)
            q_next = torch.min(
                tgt_r.gather(2, na_r).squeeze(-1), dim=0)[0]
            td_target = rr.squeeze() + self.gamma * (1 - term_b.squeeze()) * q_next
            td_target = td_target.clamp(-50.0, 50.0)  # 防止 bootstrap 爆炸
        td_loss = F.mse_loss(q_r, td_target)
        q_r_mean = q_r_all.mean(0)
        log_sum_exp_q = torch.logsumexp(q_r_mean, dim=-1)
        q_data = (q_r_mean * aa_onehot).sum(-1)
        cql_penalty = (log_sum_exp_q - q_data).mean()
        q_r_loss = td_loss + self.alpha_cql * cql_penalty
        self.opt_qr.zero_grad()
        q_r_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for net in self.q_r_nets for p in net.parameters()], 10.0)
        self.opt_qr.step()

        # --- 2. Cost Q；本段内只前向 q_c_nets(ss) 一次 ---
        q_c_all = torch.stack([net(ss) for net in self.q_c_nets], dim=0)
        q_c = q_c_all.gather(2, aa_idx_b).squeeze(-1).mean(0)
        with torch.no_grad():
            next_a = self.actor(ss_next).argmax(-1)
            tgt_c = torch.stack([t(ss_next) for t in self.q_c_targets], dim=0)
            na_c = next_a.unsqueeze(0).unsqueeze(-1).expand(k, -1, 1)
            q_c_next = torch.max(
                tgt_c.gather(2, na_c).squeeze(-1), dim=0)[0]
            c_td_target = cc.squeeze() + self.gamma * (1 - term_b.squeeze()) * q_c_next
            c_td_target = c_td_target.clamp(0.0, 10.0)
        mse_c_loss = F.mse_loss(q_c, c_td_target)
        q_c_mean = q_c_all.mean(0)
        log_sum_exp_qc = torch.logsumexp(q_c_mean, dim=-1)
        qc_data = (q_c_mean * aa_onehot).sum(-1)
        cost_cql_penalty = (log_sum_exp_qc - qc_data).mean()
        alpha_cost = self.alpha_cql
        q_c_loss = mse_c_loss + alpha_cost * cost_cql_penalty
        self.opt_qc.zero_grad()
        q_c_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for net in self.q_c_nets for p in net.parameters()], 10.0)
        self.opt_qc.step()

        # --- 3. Lagrange: min_λ λ·(J_C - ε)，clip 防止 λ 爆炸 ---
        # critic 已 step，需对 Q_C 再前向一次
        with torch.no_grad():
            pi_probs = self.actor(ss)
            q_c_pi = torch.stack([net(ss)
                                 for net in self.q_c_nets], dim=0).mean(0)
            current_risk = (pi_probs * q_c_pi).sum(-1).mean()
        # 对偶上升: Cost>ε 时 λ 应增大。min -λ(risk-ε) 等价于 max λ(risk-ε)
        lambda_loss = -self.log_lambda * (current_risk - self.cost_limit)
        self.opt_lambda.zero_grad()
        lambda_loss.backward()
        self.opt_lambda.step()
        with torch.no_grad():
            # λ ∈ [0.01, 10]，避免过度保守
            self.log_lambda.clamp_(np.log(0.01), np.log(10.0))

        # --- 4. Actor: max_π (Q_R - λ·Q_C) ---
        pi_probs = self.actor(ss)
        q_r_pi = (torch.stack([net(ss) for net in self.q_r_nets], dim=0).mean(
            0) * pi_probs).sum(-1)
        q_c_pi = (torch.stack([net(ss) for net in self.q_c_nets], dim=0).mean(
            0) * pi_probs).sum(-1)
        lam = torch.exp(self.log_lambda).clamp(0.01, 10.0).detach()
        actor_loss = -(q_r_pi - lam * q_c_pi).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self._step += 1
        # 软目标更新，避免 2000 步时突变导致 Q 爆炸
        if self._step % 10 == 0:
            for i in range(self.n_critics):
                for p_t, p in zip(self.q_r_targets[i].parameters(), self.q_r_nets[i].parameters()):
                    p_t.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
                for p_t, p in zip(self.q_c_targets[i].parameters(), self.q_c_nets[i].parameters()):
                    p_t.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        if not return_metrics:
            return None
        lam = torch.exp(self.log_lambda).clamp(0.01, 10.0)
        return {
            "q_r_loss": float(q_r_loss.detach().cpu()),
            "q_c_loss": float(q_c_loss.detach().cpu()),
            "q_c_mse": float(mse_c_loss.detach().cpu()),
            "cql_cost_penalty": float(cost_cql_penalty.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "lambda": float(lam.detach().cpu()),
            "current_risk": float(current_risk.detach().cpu()),
        }

    def train(self, data_path: str, n_steps=200000, batch_size=256, save_path="safe_cql_model.pt", log_lambda_every=0):
        """log_lambda_every>0: 每 N 步保存 lambda/risk 到 {save_path}.lambda.json，供 Figure A"""
        from data.buffer import ReplayBuffer
        buf = ReplayBuffer(data_path)
        n = len(buf)
        print(f"Training Safe CQL: {n} transitions, cost_rate={buf.c.mean():.4f}")

        s_t = torch.FloatTensor(buf.s).to(self.device)
        a_t = torch.LongTensor(buf.a).to(self.device)
        r_t = torch.FloatTensor(buf.r).unsqueeze(1).to(self.device)
        c_t = torch.FloatTensor(buf.c).unsqueeze(1).to(self.device)
        s_next_t = torch.FloatTensor(buf.s_next).to(self.device)
        term = torch.FloatTensor(buf.done | buf.timeout).unsqueeze(1).to(self.device)

        lambda_history = [] if log_lambda_every > 0 else None
        log_path = None
        if lambda_history is not None:
            log_path = Path(save_path).with_name(
                Path(save_path).stem + "_lambda.json")

        def sample():
            idx = torch.randint(
                0, n, (batch_size,), device=self.device, dtype=torch.long)
            return s_t[idx], a_t[idx], r_t[idx], c_t[idx], s_next_t[idx], term[idx]

        for step in range(n_steps):
            batch = sample()
            need_log = lambda_history is not None and log_lambda_every > 0 and (
                (step + 1) % log_lambda_every == 0)
            need_print = (step + 1) % 10000 == 0
            losses = self.update(
                *batch, return_metrics=need_log or need_print)
            if need_log:
                lambda_history.append({
                    "step": step + 1, "lambda": losses["lambda"],
                    "current_risk": losses.get("current_risk", 0),
                    "qr_loss": losses["q_r_loss"], "qc_loss": losses["q_c_loss"],
                    "q_c_mse": losses.get("q_c_mse", 0),
                    "cql_cost_penalty": losses.get("cql_cost_penalty", 0),
                })
                if log_path is not None:
                    import json
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"cost_limit": self.cost_limit, "history": lambda_history},
                            f, indent=0,
                        )
            if need_print:
                risk = losses.get("current_risk", 0)
                print(f"  step {step+1}: λ={losses['lambda']:.2f} Q_C(π)={risk:.4f} (limit={self.cost_limit}) "
                      f"qr_loss={losses['q_r_loss']:.4f} qc_loss={losses['q_c_loss']:.4f} "
                      f"qc_mse={losses.get('q_c_mse', 0):.4f} cql_c={losses.get('cql_cost_penalty', 0):.4f}")

        torch.save({"actor": self.actor.state_dict(), "q_r": self.q_r_nets[0].state_dict(),
                    "q_c": self.q_c_nets[0].state_dict()}, save_path)
        print(f"Saved {save_path}")
        if lambda_history:
            import json
            ckpt = Path(save_path)
            log_path = ckpt.with_name(ckpt.stem + "_lambda.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({"cost_limit": self.cost_limit, "history": lambda_history}, f, indent=0)
            print(f"Saved lambda history: {log_path.resolve()}")
        return self.actor

    def _action_to_idx(self, action_value: float) -> int:
        a = float(action_value)
        for i, v in enumerate(self.action_values):
            if abs(v - a) < 1e-5:
                return i
        raise ValueError(f"action {action_value} not in {self.action_values}")

    def predict_qc(self, state_raw, action_value: float) -> float:
        """当前状态下对所选动作的 Cost critic 均值 Q_C(s,a)（用于与真实二值 cost 对比）。"""
        from env.chemo_env import normalize_state

        idx = self._action_to_idx(action_value)
        s = np.array(state_raw, dtype=np.float32)
        s_norm = normalize_state(s)
        if s_norm.ndim == 1:
            s_norm = s_norm.reshape(1, -1)
        x = torch.FloatTensor(s_norm).to(self.device)
        with torch.no_grad():
            qcs = [net(x)[0, idx].item() for net in self.q_c_nets]
        return float(np.mean(qcs))

    def get_policy(self, path=None):
        """Return policy function s (raw) -> a for rollout."""
        if path:
            ckpt = torch.load(path, map_location="cpu")
            self.actor.load_state_dict(ckpt["actor"])
            if "q_c" in ckpt:
                self.q_c_nets[0].load_state_dict(ckpt["q_c"])
        self.actor.eval()
        for net in self.q_c_nets:
            net.eval()

        from env.chemo_env import normalize_state

        def policy(s):
            s_np = np.array(s, dtype=np.float32)
            s_norm = normalize_state(s_np)
            if s_norm.ndim == 1:
                s_norm = s_norm.reshape(1, -1)
            with torch.no_grad():
                x = torch.as_tensor(
                    s_norm, dtype=torch.float32, device=self.device)
                a_idx = self.actor(x).argmax(-1).item()
            return float(self.action_values[a_idx])

        return policy
