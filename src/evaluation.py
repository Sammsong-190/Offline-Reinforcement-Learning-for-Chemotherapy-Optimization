"""
统一评估框架: Agent 接口 + Evaluator
策略模式: 所有算法包装成 get_action(state) -> action
"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, List, Union
from abc import ABC, abstractmethod

from env.chemo_env import (
    step_ode, reward_fn, DEFAULT_PARAMS, MAX_STEPS, X0,
    T_CLEAR, I_SAFE, N_SAFE, ACTION_SPACE, normalize_state,
    DT, termination_info, transition_cost,
)

# 与 termination_info 对齐：这些原因视为「死亡」，survival_pct=0；timeout / cured 等为 100%
_TERMINAL_DEATH = frozenset(
    {
        "cancer_death",
        "toxicity_death",
        "organ_failure",
        "immune_collapse",
        "state_explosion",
    }
)
TERMINAL_DEATH_REASONS = _TERMINAL_DEATH


def _survival_pct(termination_reason: str) -> float:
    return 0.0 if termination_reason in _TERMINAL_DEATH else 100.0


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
        raw = self.model.predict(s_norm.reshape(1, -1))
        arr = np.asarray(raw)
        if arr.ndim >= 1 and arr.shape[-1] > 1:
            idx = int(np.argmax(arr, axis=-1).ravel()[0])
        else:
            idx = int(arr.ravel()[0])
        return float(np.asarray(ACTION_SPACE[idx]).item())


class PyTorchAgent(Agent):
    """PyTorch 模型 (BC, Safe CQL) 包装为统一接口"""

    def __init__(self, path: str, agent_type: str = "safe_cql"):
        self.agent_type = agent_type
        if agent_type == "safe_cql":
            from src.algos.safe_cql import SafeCQL
            self._algo = SafeCQL()
            self._policy = self._algo.get_policy(path)

            def predict_qc(state, action_value: float):
                return self._algo.predict_qc(state, action_value)

            self.predict_qc = predict_qc
        elif agent_type == "bc":
            import torch
            from src.bc_policy import PolicyNet
            net = PolicyNet()
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()

            def policy(s):
                s_norm = normalize_state(s)
                idx = net(torch.FloatTensor(
                    s_norm).unsqueeze(0)).argmax(1).item()
                return float(ACTION_SPACE[idx])
            self._policy = policy
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    def get_action(self, state):
        return self._policy(state)


def _rollout_one(
    agent: Agent,
    params=None,
    seed=None,
    randomize_patient=False,
    patient_ctx: Optional[Union[Dict, Callable[[], Dict]]] = None,
) -> dict:
    """
    单次 rollout。务必传入 patient_ctx（dict 或每局可调用的 factory），
    以便用队列特异 i_safe / c_tox / n_safe 判定终止与违规；None 时与旧版 is_done 一致。
    """
    if seed is not None:
        from env.robust import set_seed
        set_seed(seed)
    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    params = params or DEFAULT_PARAMS
    if randomize_patient and patient_ctx is None:
        from env.patient import randomize_params
        params = randomize_params(params, scale=0.1)

    if callable(patient_ctx):
        ctx = patient_ctx()
    else:
        ctx = patient_ctx

    if ctx is not None:
        dyn_params = ctx.get("params", params)
        sde_sigma = float(ctx.get("sde_sigma", 0.0))
    else:
        dyn_params = params
        sde_sigma = 0.0

    x = np.array(X0, dtype=np.float32)
    R, actions, violation_steps = 0.0, [], 0
    qc_preds = []
    tumor_start = float(X0[1])
    survival_steps = MAX_STEPS
    termination_reason = "timeout"

    for step in range(MAX_STEPS):
        x_prev = x.copy()
        a = agent.get_action(x)
        actions.append(float(a))
        if hasattr(agent, "predict_qc"):
            try:
                qc_preds.append(agent.predict_qc(x_prev, a))
            except Exception:
                qc_preds.append(float("nan"))
        x = step_ode(x, a, DT, dyn_params, sde_sigma=sde_sigma, rng=rng)
        R += reward_fn(x, DT, s_prev=x_prev)
        if ctx is not None:
            violation_steps += int(transition_cost(x,
                                   ctx["i_safe"], ctx["n_safe"]))
        else:
            violation_steps += int(transition_cost(x))
        done, reason = termination_info(x, ctx)
        if done:
            survival_steps = step + 1
            termination_reason = reason
            break

    total_steps = survival_steps
    drug_total = np.sum(actions) * DT if actions else 1e-8
    survival_time = float(survival_steps) * float(DT)
    cured = termination_reason == "cured"
    true_cost_rate = violation_steps / total_steps if total_steps else 0.0
    out = {
        "return": R,
        "tumor_clear_pct": 100.0 if cured else 0.0,
        "survival_pct": _survival_pct(termination_reason),
        "avg_dose": np.mean(actions) if actions else 0.0,
        "constraint_violation_rate_pct": true_cost_rate * 100.0,
        "true_cost_rate": float(true_cost_rate),
        "treatment_efficiency": (tumor_start - x[1]) / drug_total if drug_total > 0 else 0.0,
        "survival_steps": int(survival_steps),
        "survival_time": survival_time,
        "termination_reason": termination_reason,
    }
    if qc_preds:
        out["mean_qc_predicted"] = float(np.nanmean(qc_preds))
        out["max_qc_predicted"] = float(np.nanmax(qc_preds))
    return out


class Evaluator:
    """统一评估器: evaluate_all(agents, n_episodes, seeds) -> metrics, save_csv"""

    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS

    def evaluate_agent(
        self,
        agent: Agent,
        n_episodes: int = 20,
        seeds: Optional[List[int]] = None,
        randomize_patient: bool = False,
        patient_ctx: Optional[Union[Dict, Callable[[], Dict]]] = None,
        cohort_sample: bool = False,
    ) -> dict:
        from env.patient_cohorts import PatientGenerator

        seeds = seeds or [42]
        all_metrics = []
        for i, seed in enumerate(seeds):
            for ep in range(n_episodes):
                ep_seed = seed + ep * 1000 + i if randomize_patient else seed

                if cohort_sample:
                    _ep = ep_seed

                    def sampled_ctx():
                        return PatientGenerator(self.params, rng=np.random.default_rng(_ep + 4242)).sample(
                            jitter=0.0
                        )

                    ctx_arg = sampled_ctx
                else:
                    ctx_arg = patient_ctx

                m = _rollout_one(
                    agent,
                    self.params,
                    seed=ep_seed,
                    randomize_patient=randomize_patient and not cohort_sample,
                    patient_ctx=ctx_arg,
                )
                all_metrics.append(m)

        reasons = list({x["termination_reason"] for x in all_metrics})
        base = {
            "return_mean": float(np.mean([x["return"] for x in all_metrics])),
            "return_std": float(np.std([x["return"] for x in all_metrics])),
            "tumor_clear_pct": float(np.mean([x["tumor_clear_pct"] for x in all_metrics])),
            "survival_pct": float(np.mean([x["survival_pct"] for x in all_metrics])),
            "avg_dose": float(np.mean([x["avg_dose"] for x in all_metrics])),
            "constraint_violation_rate_pct": float(np.mean([x["constraint_violation_rate_pct"] for x in all_metrics])),
            "treatment_efficiency": float(np.mean([x["treatment_efficiency"] for x in all_metrics])),
            "survival_steps_mean": float(np.mean([x["survival_steps"] for x in all_metrics])),
            "survival_steps_std": float(np.std([x["survival_steps"] for x in all_metrics])),
            "survival_time_mean": float(np.mean([x["survival_time"] for x in all_metrics])),
        }
        for r in reasons:
            base[f"frac_{r}"] = float(
                np.mean([x["termination_reason"] == r for x in all_metrics]))
        if all_metrics and "mean_qc_predicted" in all_metrics[0]:
            base["mean_qc_predicted"] = float(
                np.nanmean([x.get("mean_qc_predicted", np.nan) for x in all_metrics]))
            base["mean_qc_predicted_std"] = float(
                np.nanstd([x.get("mean_qc_predicted", np.nan) for x in all_metrics]))
        return base

    def episode_rollouts(
        self,
        agent: Agent,
        n_episodes: int = 50,
        base_seed: int = 42,
        randomize_patient: bool = False,
        patient_ctx: Optional[Union[Dict, Callable[[], Dict]]] = None,
    ) -> List[dict]:
        """逐条 episode 指标（用于 QC mismatch 小提琴图 / 散点）。"""
        rows = []
        for ep in range(n_episodes):
            ep_seed = base_seed + ep * 1000
            m = _rollout_one(
                agent,
                self.params,
                seed=ep_seed,
                randomize_patient=randomize_patient,
                patient_ctx=patient_ctx,
            )
            m["episode"] = ep
            rows.append(m)
        return rows

    def evaluate_all(
        self,
        agents: Dict[str, Agent],
        n_episodes: int = 20,
        seeds: Optional[List[int]] = None,
        randomize_patient: bool = False,
        patient_ctx: Optional[Union[Dict, Callable[[], Dict]]] = None,
        cohort_sample: bool = False,
    ) -> dict:
        """agents: {name: Agent}"""
        results = {}
        for name, agent in agents.items():
            results[name] = self.evaluate_agent(
                agent, n_episodes, seeds, randomize_patient, patient_ctx, cohort_sample
            )
        return results

    @staticmethod
    def save_csv(results: dict, path: str):
        """保存为 CSV，供 notebooks 画 Return vs Cost / KM 相关列"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, m in results.items():
            row = {
                "policy": name,
                "return_mean": m["return_mean"],
                "return_std": m["return_std"],
                "constraint_violation_rate_pct": m["constraint_violation_rate_pct"],
                "survival_pct": m["survival_pct"],
                "tumor_clear_pct": m["tumor_clear_pct"],
                "avg_dose": m["avg_dose"],
                "treatment_efficiency": m["treatment_efficiency"],
            }
            if "survival_steps_mean" in m:
                row["survival_steps_mean"] = m["survival_steps_mean"]
                row["survival_time_mean"] = m["survival_time_mean"]
            for k, v in m.items():
                if k.startswith("frac_"):
                    row[k] = v
            for k in ("mean_qc_predicted", "mean_qc_predicted_std"):
                if k in m:
                    row[k] = m[k]
            rows.append(row)
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames,
                               extrasaction="ignore", restval="")
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
    for p in (root / "checkpoints" / "safe_cql_limit0.1_seed42.pt", root / "safe_cql_model.pt"):
        if p.exists():
            agents["SafeCQL"] = PyTorchAgent(str(p), "safe_cql")
            break
    if (root / "cql_model.d3").exists():
        try:
            agents["CQL"] = D3RLPyAgent(str(root / "cql_model.d3"))
        except Exception:
            pass
    for dose in [0.0, 0.5, 1.0, 2.0]:
        agents[f"Fixed{dose}"] = FixedDoseAgent(dose)
    return agents
