"""
Safe CQL: Lagrangian Constrained Offline RL
CMDP: max E[sum r] s.t. E[sum c] <= epsilon

论文算法对应: src/algos/safe_cql.py 中的 update() 与 Lagrangian Loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algos.base import BaseAlgo
from src.models import Actor, Critic, SafetyCritic


class SafeCQL(BaseAlgo):
    """
    Lagrangian Safe CQL.
    update() 包含: Reward Q (CQL), Cost Q, Lagrange, Actor 四步。
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.alpha_cql = alpha_cql
        self.cost_limit = cost_limit
        self.target_update_interval = target_update_interval
        self.n_critics = n_critics
        self.n_actions = n_actions
        self.action_values = [0.0, 0.5, 1.0, 2.0]
        self._step = 0

        self.actor = Actor(state_dim, n_actions, hidden).to(self.device)
        self.q_r_nets = [Critic(state_dim, n_actions, hidden).to(self.device) for _ in range(n_critics)]
        self.q_r_targets = [Critic(state_dim, n_actions, hidden).to(self.device) for _ in range(n_critics)]
        self.q_c_nets = [SafetyCritic(state_dim, n_actions, hidden).to(self.device) for _ in range(n_critics)]
        self.q_c_targets = [SafetyCritic(state_dim, n_actions, hidden).to(self.device) for _ in range(n_critics)]
        for i in range(n_critics):
            self.q_r_targets[i].load_state_dict(self.q_r_nets[i].state_dict())
            self.q_c_targets[i].load_state_dict(self.q_c_nets[i].state_dict())

        self.log_lambda = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.opt_qr = torch.optim.Adam([p for net in self.q_r_nets for p in net.parameters()], lr=critic_lr)
        self.opt_qc = torch.optim.Adam([p for net in self.q_c_nets for p in net.parameters()], lr=critic_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_lambda = torch.optim.Adam([self.log_lambda], lr=lagrangian_lr)

    def update(self, ss, aa, rr, cc, ss_next, term_b):
        """
        单步更新。对应论文中的 Lagrangian Loss:
        L = L_CQL(Q_R) + L_TD(Q_C) + L_λ(λ) + L_π(π)
        """
        aa_onehot = F.one_hot(aa, self.n_actions).float()

        # --- 1. Reward Q (CQL conservative) ---
        q_r = sum(net(ss).gather(1, aa.unsqueeze(1)).squeeze(1) for net in self.q_r_nets) / self.n_critics
        with torch.no_grad():
            next_a = self.actor(ss_next).argmax(-1)
            q_next = torch.min(torch.stack([t(ss_next).gather(1, next_a.unsqueeze(1)).squeeze(1) for t in self.q_r_targets]), dim=0)[0]
            td_target = rr.squeeze() + self.gamma * (1 - term_b.squeeze()) * q_next
        td_loss = F.mse_loss(q_r, td_target)
        log_sum_exp_q = torch.logsumexp(torch.stack([net(ss) for net in self.q_r_nets], dim=0).mean(0), dim=-1)
        q_data = (torch.stack([net(ss) for net in self.q_r_nets], dim=0).mean(0) * aa_onehot).sum(-1)
        cql_penalty = (log_sum_exp_q - q_data).mean()
        q_r_loss = td_loss + self.alpha_cql * cql_penalty
        self.opt_qr.zero_grad()
        q_r_loss.backward()
        self.opt_qr.step()

        # --- 2. Cost Q (MSE) ---
        q_c = sum(net(ss).gather(1, aa.unsqueeze(1)).squeeze(1) for net in self.q_c_nets) / self.n_critics
        with torch.no_grad():
            next_a = self.actor(ss_next).argmax(-1)
            q_c_next = torch.min(torch.stack([t(ss_next).gather(1, next_a.unsqueeze(1)).squeeze(1) for t in self.q_c_targets]), dim=0)[0]
            c_td_target = cc.squeeze() + self.gamma * (1 - term_b.squeeze()) * q_c_next
        q_c_loss = F.mse_loss(q_c, c_td_target)
        self.opt_qc.zero_grad()
        q_c_loss.backward()
        self.opt_qc.step()

        # --- 3. Lagrange: min_λ λ·(J_C - ε) ---
        with torch.no_grad():
            pi_probs = self.actor(ss)
            q_c_pi = torch.stack([net(ss) for net in self.q_c_nets], dim=0).mean(0)
            current_risk = (pi_probs * q_c_pi).sum(-1).mean()
        lambda_loss = self.log_lambda * (current_risk - self.cost_limit)
        self.opt_lambda.zero_grad()
        lambda_loss.backward()
        self.opt_lambda.step()

        # --- 4. Actor: max_π (Q_R - λ·Q_C) ---
        pi_probs = self.actor(ss)
        q_r_pi = (torch.stack([net(ss) for net in self.q_r_nets], dim=0).mean(0) * pi_probs).sum(-1)
        q_c_pi = (torch.stack([net(ss) for net in self.q_c_nets], dim=0).mean(0) * pi_probs).sum(-1)
        lam = torch.exp(self.log_lambda).clamp(0.01, 100.0).detach()
        actor_loss = -(q_r_pi - lam * q_c_pi).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self._step += 1
        if self._step % self.target_update_interval == 0:
            for i in range(self.n_critics):
                self.q_r_targets[i].load_state_dict(self.q_r_nets[i].state_dict())
                self.q_c_targets[i].load_state_dict(self.q_c_nets[i].state_dict())

        return {
            "q_r_loss": q_r_loss.item(),
            "q_c_loss": q_c_loss.item(),
            "actor_loss": actor_loss.item(),
            "lambda": torch.exp(self.log_lambda).item(),
        }

    def train(self, data_path: str, n_steps=200000, batch_size=256, save_path="safe_cql_model.pt"):
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

        def sample():
            idx = np.random.choice(n, size=batch_size, replace=True)
            return s_t[idx], a_t[idx], r_t[idx], c_t[idx], s_next_t[idx], term[idx]

        for step in range(n_steps):
            batch = sample()
            losses = self.update(*batch)
            if (step + 1) % 10000 == 0:
                print(f"  step {step+1}: lambda={losses['lambda']:.3f} q_r={losses['q_r_loss']:.4f}")

        torch.save({"actor": self.actor.state_dict(), "q_r": self.q_r_nets[0].state_dict(), "q_c": self.q_c_nets[0].state_dict()}, save_path)
        print(f"Saved {save_path}")
        return self.actor

    def get_policy(self, path=None):
        """Return policy function s (raw) -> a for rollout."""
        if path:
            ckpt = torch.load(path, map_location="cpu")
            self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

        from env.chemo_env import normalize_state

        def policy(s):
            s_np = np.array(s, dtype=np.float32)
            s_norm = normalize_state(s_np)
            if s_norm.ndim == 1:
                s_norm = s_norm.reshape(1, -1)
            with torch.no_grad():
                a_idx = self.actor(torch.FloatTensor(s_norm)).argmax(-1).item()
            return float(self.action_values[a_idx])

        return policy
