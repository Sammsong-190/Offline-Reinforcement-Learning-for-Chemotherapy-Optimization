"""
统一评估框架: Agent 接口 + Evaluator
策略模式: 所有算法包装成 get_action(state) -> action
"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, List
from abc import ABC, abstractmethod

from env.chemo_env import (
    step_ode, reward_fn_v3, DEFAULT_PARAMS, MAX_STEPS, X0, is_done,
    T_CLEAR, I_SAFE, N_SAFE, ACTION_SPACE, normalize_state,
)


class Agent(ABC):
    """Agent 接口: get_action(state) -> action (float)"""

    @abstractmethod
    def get_action(self, state: np.ndarray) -> float:
        pass

    def __call__(self, state: np.ndarray) -> float:
        return self.get_action(state)


class ExpertAgent(Agent):
    def __init__(self, epsilon=0.0):
        from data.generate import expert_policy
        self._policy = lambda s: expert_policy(s, epsilon=epsilon)

    def get_action(self, state):
        return self._policy(state)


class RandomAgent(Agent):
    def get_action(self, state):
        return float(np.random.choice(ACTION_SPACE))


class FixedDoseAgent(Agent):
    def __init__(self, dose: float):
        self.dose = dose

    def get_action(self, state):
        return self.dose


class D3RLPyAgent(Agent):
    """d3rlpy 模型包装为统一接口"""

    def __init__(self, path: str):
        import d3rlpy
        self.model = d3rlpy.load_learnable(path)

    def get_action(self, state):
        s_norm = normalize_state(state)
        idx = self.model.predict(s_norm.reshape(1, -1))
        return float(ACTION_SPACE[int(idx)])


class PyTorchAgent(Agent):
    """PyTorch 模型 (BC, Safe CQL) 包装为统一接口"""

    def __init__(self, path: str, agent_type: str = "safe_cql"):
        self.agent_type = agent_type
        if agent_type == "safe_cql":
            from src.algos.safe_cql import SafeCQL
            algo = SafeCQL()
            self._policy = algo.get_policy(path)
        elif agent_type == "bc":
            import torch
            from train_offline import PolicyNet
            net = PolicyNet()
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()

            def policy(s):
                s_norm = normalize_state(s)
                idx = net(torch.FloatTensor(s_norm).unsqueeze(0)).argmax(1).item()
                return float(ACTION_SPACE[idx])
            self._policy = policy
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    def get_action(self, state):
        return self._policy(state)


def _rollout_one(agent: Agent, params=None, seed=None, randomize_patient=False) -> dict:
    """单次 rollout，返回 metrics dict。randomize_patient 增加 OOD 多样性。"""
    if seed is not None:
        from env.robust import set_seed
        set_seed(seed)
    params = params or DEFAULT_PARAMS
    if randomize_patient:
        from env.patient import randomize_params
        params = randomize_params(params, scale=0.1)
    x = np.array(X0, dtype=np.float32)
    R, actions, violation_steps = 0.0, [], 0
    tumor_start = X0[1]
    for step in range(MAX_STEPS):
        x_prev = x.copy()
        a = agent.get_action(x)
        actions.append(float(a))
        x = step_ode(x, a, 0.3, params)
        R += reward_fn_v3(x, 0.3, s_prev=x_prev)
        if x[2] < I_SAFE or x[0] < N_SAFE:
            violation_steps += 1
        if is_done(x):
            break
    total_steps = step + 1
    drug_total = np.sum(actions) * 0.3 if actions else 1e-8
    return {
        "return": R,
        "tumor_clear_pct": 100.0 if x[1] < T_CLEAR else 0.0,
        "survival_pct": 100.0 if (x[0] > 0.1 and x[2] > 0.1) else 0.0,
        "avg_dose": np.mean(actions) if actions else 0.0,
        "constraint_violation_rate_pct": violation_steps / total_steps * 100,
        "treatment_efficiency": (tumor_start - x[1]) / drug_total if drug_total > 0 else 0.0,
    }


class Evaluator:
    """统一评估器: evaluate_all(agents, n_episodes, seeds) -> metrics, save_csv"""

    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS

    def evaluate_agent(self, agent: Agent, n_episodes: int = 20, seeds: Optional[List[int]] = None,
                      randomize_patient: bool = False) -> dict:
        seeds = seeds or [42]
        all_metrics = []
        for i, seed in enumerate(seeds):
            for ep in range(n_episodes):
                # 每 episode 不同 seed 以增加多样性 (尤其 deterministic policy)
                ep_seed = seed + ep * 1000 + i if randomize_patient else seed
                m = _rollout_one(agent, self.params, seed=ep_seed, randomize_patient=randomize_patient)
                all_metrics.append(m)
        return {
            "return_mean": np.mean([x["return"] for x in all_metrics]),
            "return_std": np.std([x["return"] for x in all_metrics]),
            "tumor_clear_pct": np.mean([x["tumor_clear_pct"] for x in all_metrics]),
            "survival_pct": np.mean([x["survival_pct"] for x in all_metrics]),
            "avg_dose": np.mean([x["avg_dose"] for x in all_metrics]),
            "constraint_violation_rate_pct": np.mean([x["constraint_violation_rate_pct"] for x in all_metrics]),
            "treatment_efficiency": np.mean([x["treatment_efficiency"] for x in all_metrics]),
        }

    def evaluate_all(self, agents: Dict[str, Agent], n_episodes: int = 20,
                     seeds: Optional[List[int]] = None, randomize_patient: bool = False) -> dict:
        """agents: {name: Agent}"""
        results = {}
        for name, agent in agents.items():
            results[name] = self.evaluate_agent(agent, n_episodes, seeds, randomize_patient)
        return results

    @staticmethod
    def save_csv(results: dict, path: str):
        """保存为 CSV，供 notebooks 画 Return vs Cost 图"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, m in results.items():
            rows.append({
                "policy": name,
                "return_mean": m["return_mean"],
                "return_std": m["return_std"],
                "constraint_violation_rate_pct": m["constraint_violation_rate_pct"],
                "survival_pct": m["survival_pct"],
                "tumor_clear_pct": m["tumor_clear_pct"],
                "avg_dose": m["avg_dose"],
                "treatment_efficiency": m["treatment_efficiency"],
            })
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {path}")


def build_agents(root: Path) -> Dict[str, Agent]:
    """根据已有模型构建 agents dict。缺失的跳过。"""
    agents = {}
    agents["Expert"] = ExpertAgent()
    agents["Random"] = RandomAgent()
    if (root / "bc_policy.pt").exists():
        agents["BC"] = PyTorchAgent(str(root / "bc_policy.pt"), "bc")
    if (root / "safe_cql_model.pt").exists():
        agents["SafeCQL"] = PyTorchAgent(str(root / "safe_cql_model.pt"), "safe_cql")
    if (root / "cql_model.d3").exists():
        try:
            agents["CQL"] = D3RLPyAgent(str(root / "cql_model.d3"))
        except Exception:
            pass
    if (root / "iql_model.d3").exists():
        try:
            agents["IQL"] = D3RLPyAgent(str(root / "iql_model.d3"))
        except Exception:
            pass
    for dose in [0.0, 0.5, 1.0, 2.0]:
        agents[f"Fixed{dose}"] = FixedDoseAgent(dose)
    return agents
