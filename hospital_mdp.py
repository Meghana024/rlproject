"""
Hospital Resource Management MDP Engine
Markov Decision Process & Value Iteration Solver for Patient Admission and Resource Allocation.
"""

from dataclasses import dataclass, field
import itertools
import math
import csv
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class MDPParams:
    gamma: float = 0.90
    theta: float = 1e-6
    max_iterations: int = 1000
    cap: int = 2
    
    # Arrival probabilities (Bernoulli: (value, prob))
    crit_arr_p: float = 0.25
    norm_arr_p: float = 0.55
    
    # Discharge probabilities per occupied bed
    crit_dis_p: float = 0.20
    norm_dis_p: float = 0.15
    
    # Rewards & penalties
    reward_crit: float = 40.0
    reward_norm: float = 10.0
    queue_crit_penalty: float = 12.0
    queue_norm_penalty: float = 3.0
    bed_gen_cost: float = 1.0
    bed_icu_cost: float = 0.5
    queue_full_crit_penalty: float = 10.0
    queue_full_norm_penalty: float = 3.0
    terminal_overload_penalty: float = 100.0


ACTION_NAMES = {
    0: "Wait",
    1: "Admit Critical",
    2: "Admit Non-Critical"
}

ACTION_ICONS = {
    0: "⏳ Wait",
    1: "🚨 Admit Critical",
    2: "🩺 Admit Non-Critical"
}


def get_all_states(cap: int = 2) -> List[Tuple[int, ...]]:
    return list(itertools.product(range(cap + 1), repeat=6))


def get_initial_state(cap: int = 2) -> Tuple[int, ...]:
    return (1, 1, cap, cap, cap, cap)


def get_terminal_state(cap: int = 2) -> Tuple[int, ...]:
    return (cap, cap, 0, 0, 0, 0)


def valid_actions(s: Tuple[int, ...]) -> List[int]:
    """Returns valid action IDs for a given state (Qc, Qn, Bg, Bi, V, S)."""
    Qc, Qn, Bg, Bi, V, S = s
    actions = [0]
    if Qc > 0 and Bi > 0 and V > 0 and S > 0:
        actions.append(1)
    if Qn > 0 and Bg > 0 and S > 0:
        actions.append(2)
    return actions


def discharge_probabilities(occupied: int, p: float) -> List[Tuple[int, float]]:
    """Binomial distribution for discharges given occupied beds and discharge probability p."""
    result = []
    for d in range(occupied + 1):
        prob = math.comb(occupied, d) * (p ** d) * ((1 - p) ** (occupied - d))
        result.append((d, prob))
    return result


def compute_transitions(s: Tuple[int, ...], action: int, params: MDPParams) -> Dict[Tuple[int, ...], float]:
    """Computes next state probabilities P(s' | s, a)."""
    Qc, Qn, Bg, Bi, V, S = s
    cap = params.cap
    
    if action == 1:
        Qc -= 1
        Bi -= 1
        V -= 1
        S -= 1
    elif action == 2:
        Qn -= 1
        Bg -= 1
        S -= 1

    occupied_critical = cap - Bi
    occupied_normal = cap - Bg

    critical_arrivals = [(0, 1.0 - params.crit_arr_p), (1, params.crit_arr_p)]
    normal_arrivals = [(0, 1.0 - params.norm_arr_p), (1, params.norm_arr_p)]

    critical_discharges = discharge_probabilities(occupied_critical, params.crit_dis_p)
    normal_discharges = discharge_probabilities(occupied_normal, params.norm_dis_p)

    result: Dict[Tuple[int, ...], float] = {}

    for ca, p_ca in critical_arrivals:
        for na, p_na in normal_arrivals:
            for cd, p_cd in critical_discharges:
                for nd, p_nd in normal_discharges:
                    new_Qc = min(cap, Qc + ca)
                    new_Qn = min(cap, Qn + na)
                    new_Bg = min(cap, Bg + nd)
                    new_Bi = min(cap, Bi + cd)
                    new_V = min(cap, V + cd)
                    new_S = min(cap, S + cd + nd)

                    ns = (new_Qc, new_Qn, new_Bg, new_Bi, new_V, new_S)
                    prob = p_ca * p_na * p_cd * p_nd
                    result[ns] = result.get(ns, 0.0) + prob

    return result


def compute_reward(s: Tuple[int, ...], action: int, ns: Tuple[int, ...], params: MDPParams) -> float:
    """Computes the reward R(s, a, s')."""
    Qc, Qn, Bg, Bi, V, S = s
    cap = params.cap
    terminal_state = get_terminal_state(cap)

    r = 0.0
    if action == 1:
        r += params.reward_crit
    elif action == 2:
        r += params.reward_norm

    # Holding costs
    r -= params.queue_crit_penalty * Qc
    r -= params.queue_norm_penalty * Qn

    # Resource utilization costs
    r -= params.bed_gen_cost * (cap - Bg)
    r -= params.bed_icu_cost * (cap - Bi)

    # Queue overflow warnings
    if ns[0] == cap:
        r -= params.queue_full_crit_penalty
    if ns[1] == cap:
        r -= params.queue_full_norm_penalty

    # Terminal state overload penalty
    if ns == terminal_state:
        r -= params.terminal_overload_penalty

    return r


class HospitalMDP:
    def __init__(self, params: MDPParams = None):
        self.params = params or MDPParams()
        self.states = get_all_states(self.params.cap)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.initial_state = get_initial_state(self.params.cap)
        self.terminal_state = get_terminal_state(self.params.cap)
        
        # Precomputed transitions cache
        self.trans_cache: Dict[Tuple[Tuple[int, ...], int], Dict[Tuple[int, ...], float]] = {}
        self.reward_cache: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], float] = {}

    def get_transitions(self, s: Tuple[int, ...], a: int) -> Dict[Tuple[int, ...], float]:
        key = (s, a)
        if key not in self.trans_cache:
            self.trans_cache[key] = compute_transitions(s, a, self.params)
        return self.trans_cache[key]

    def get_reward(self, s: Tuple[int, ...], a: int, ns: Tuple[int, ...]) -> float:
        key = (s, a, ns)
        if key not in self.reward_cache:
            self.reward_cache[key] = compute_reward(s, a, ns, self.params)
        return self.reward_cache[key]

    def solve_value_iteration(self, callback=None) -> Tuple[np.ndarray, Dict[Tuple[int, ...], Dict[int, float]], Dict[Tuple[int, ...], int], List[float]]:
        """Executes Value Iteration until convergence."""
        num_states = len(self.states)
        V = np.zeros(num_states, dtype=np.float64)
        delta_history = []
        terminal_idx = self.state_to_idx[self.terminal_state]

        for iteration in range(1, self.params.max_iterations + 1):
            new_V = np.zeros_like(V)
            
            for s in self.states:
                if s == self.terminal_state:
                    new_V[self.state_to_idx[s]] = 0.0
                    continue

                best_val = -float("inf")
                for a in valid_actions(s):
                    q = 0.0
                    for ns, prob in self.get_transitions(s, a).items():
                        r = self.get_reward(s, a, ns)
                        q += prob * (r + self.params.gamma * V[self.state_to_idx[ns]])
                    if q > best_val:
                        best_val = q
                
                new_V[self.state_to_idx[s]] = best_val

            delta = float(np.max(np.abs(new_V - V)))
            delta_history.append(delta)
            V = new_V

            if callback:
                callback(iteration, delta)

            if delta < self.params.theta:
                break

        # Extract optimal Q-values and policy
        Q: Dict[Tuple[int, ...], Dict[int, float]] = {}
        policy: Dict[Tuple[int, ...], int] = {}

        for s in self.states:
            Q[s] = {}
            for a in valid_actions(s):
                q = 0.0
                for ns, prob in self.get_transitions(s, a).items():
                    r = self.get_reward(s, a, ns)
                    q += prob * (r + self.params.gamma * V[self.state_to_idx[ns]])
                Q[s][a] = q
            
            policy[s] = max(Q[s], key=Q[s].get)

        return V, Q, policy, delta_history

    def sample_next_state(self, s: Tuple[int, ...], a: int, rng: np.random.Generator = None) -> Tuple[int, ...]:
        trans = self.get_transitions(s, a)
        states = list(trans.keys())
        probs = np.array(list(trans.values()), dtype=np.float64)
        probs /= probs.sum()
        
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(states), p=probs)
        return states[idx]

    def simulate(self, policy: Dict[Tuple[int, ...], int], num_episodes: int = 100, max_steps: int = 50, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        episode_returns = []
        episode_lengths = []
        actions_taken = {0: 0, 1: 0, 2: 0}
        trajectory_sample = []

        for ep in range(num_episodes):
            s = self.initial_state
            G = 0.0
            steps = 0
            
            for t in range(max_steps):
                if s == self.terminal_state:
                    break
                
                a = policy.get(s, 0)
                actions_taken[a] += 1
                ns = self.sample_next_state(s, a, rng)
                r = self.get_reward(s, a, ns)
                G += (self.params.gamma ** t) * r
                
                if ep == 0:
                    trajectory_sample.append({
                        "step": t + 1,
                        "state": s,
                        "action": ACTION_NAMES[a],
                        "reward": round(r, 2),
                        "next_state": ns
                    })
                
                s = ns
                steps += 1

            episode_returns.append(G)
            episode_lengths.append(steps)

        running_avg = np.cumsum(episode_returns) / np.arange(1, num_episodes + 1)
        
        return {
            "returns": episode_returns,
            "running_avg": running_avg.tolist(),
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_length": float(np.mean(episode_lengths)),
            "action_counts": actions_taken,
            "sample_trajectory": trajectory_sample
        }
