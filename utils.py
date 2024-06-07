import sympy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import re
import optax
import jax
import distrax

from sympy.parsing.sympy_parser import parse_expr
from sympy.printing import pycode
from functools import partial
from flax import linen as nn
from random import random
from tqdm.auto import tqdm
from jax import jit
from typing import Callable

systems = {
    "SIM": {
        "system": '''
YD == W * N_s - T
T == theta * W * N_s
C == alpha_1 * YD + alpha_2 * H_h_prev
dH_s == H_s - H_s_prev
dH_s == G - T
dH_h == H_h - H_h_prev
dH_h == YD - C
Y == C + G
N_s == Y / W
dH_h == dH_s
''',
        "params": { # TODO: consider generating params randomly as  bigger models have a lot of ones
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "W": 0.05,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dH_h, dH_s"
    },

    "SIMEX": {
        "system": '''
YD == W * N_s - T
T == theta * W * N_s
C == alpha_1 * YD_e + alpha_2 * H_h_prev
dH_s == H_s - H_s_prev
dH_s == G - T
dH_h == H_h - H_h_prev
dH_h == YD - C
Y == C + G
N_s == Y / W
dH_d == H_d - H_h_prev
dH_d == YD_e - C
YD_e == YD_prev
dH_h == dH_s
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "W": 0.05,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dH_h, dH_s, dH_d"
    },


    "PC": {
        "system": '''
Y == C + G
YD == Y - T + r_prev * B_h_prev
T == theta * (Y + r_prev * B_h_prev)
V == V_prev + YD - C
C == alpha_1 * YD + alpha_2 * V_prev
H_h == V - B_h
B_h / V == lambda_0 + lambda_1 * r - lambda_2 * YD / V
dB_s == B_s - B_s_prev
dB_s == G + r_prev * B_s_prev - T - r_prev * B_cb_prev
dH_s == H_s - H_s_prev
dH_s == dB_cb
B_cb == B_s - B_h
dB_cb == B_cb - B_cb_prev
r == r_cap
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "lambda_0": 0.1,
            "lambda_1": 0.2,
            "lambda_2": 0.3,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            },
            "r_cap": {
                "transform": lambda x: jnp.clip(0.05 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dB_s, dH_s, dB_cb,"
    },

    "PCEX_deterministic": {
        "system": '''
Y == C + G
YD == Y - T + r_prev * B_h_prev
T == theta * (Y + r_prev * B_h_prev)
V == V_prev + YD - C
C == alpha_1 * YD_e + alpha_2 * V_prev
B_d / V_e == lambda_0 + lambda_1 * r - lambda_2 * YD_e / V_e
H_d / V_e == 1 - lambda_0 - lambda_1 * r + lambda_2 * YD_e / V_e
H_d == V_e - B_d
V_e == V_prev + YD_e - C
H_h == V - B_h
B_h == B_d
dB_s == B_s - B_s_prev
dB_s == G + r_prev * B_s_prev - T - r_prev * B_cb_prev
dH_s == H_s - H_s_prev
dH_s == dB_cb
dB_cb == B_cb - B_cb_prev
B_cb == B_s - B_h
r == r_cap
YD_e == YD_prev
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "lambda_0": 0.1,
            "lambda_1": 0.2,
            "lambda_2": 0.3,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            },
            "r_cap": {
                "transform": lambda x: jnp.clip(0.05 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dB_s, dH_s, dB_cb,"
    },

    "PCEX_stochastic": {
        "system": '''
Y == C + G
YD == Y - T + r_prev * B_h_prev
T == theta * (Y + r_prev * B_h_prev)
V == V_prev + YD - C
C == alpha_1 * YD_e + alpha_2 * V_prev
B_d / V_e == lambda_0 + lambda_1 * r - lambda_2 * YD_e / V_e
H_d / V_e == 1 - lambda_0 - lambda_1 * r + lambda_2 * YD_e / V_e
H_d == V_e - B_d
V_e == V_prev + YD_e - C
H_h == V - B_h
B_h == B_d
dB_s == B_s - B_s_prev
dB_s == G + r_prev * B_s_prev - T - r_prev * B_cb_prev
dH_s == H_s - H_s_prev
dH_s == dB_cb
dB_cb == B_cb - B_cb_prev
B_cb == B_s - B_h
r == r_cap
YD_e == YD_prev * (1 + Ra)
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "lambda_0": 0.1,
            "lambda_1": 0.2,
            "lambda_2": 0.3,
            "Ra_std": 0.4,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            },
            "r_cap": {
                "transform": lambda x: jnp.clip(0.05 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dB_s, dH_s, dB_cb",
        "stubs": {
            "Ra": "jax.random.normal(seed, (1,)) * Ra_std",
        }
    },

    "LP": {
        "system": '''
Y == C + G
YD_r == Y - T + r_b_prev * B_h_prev + BL_h_prev
T == theta * (Y * r_b_prev * B_h_prev + BL_h_prev)
V == V_prev + YD_r - C + CG
CG == dp_bL * BL_h_prev
C == alpha_1 * YD_r_e + alpha_2 * V_prev
V_e == V_prev + YD_r_e - C + CG_e
H_h == V - B_h - p_bL * BL_h
H_d == V_e - B_d - p_bL * BL_d
B_d / V_e == lambda_20 + lambda_22 * r_b + lambda_23 * ERr_bL + lambda_24 * YD_r_e / V_e
BL_d * p_bL / V_e == lambda_30 + lambda_32 * r_b + lambda_33 * ERr_bL + lambda_34 * YD_r_e / V_e
B_h == B_d
BL_h == BL_d
dB_s == B_s - B_s_prev
dB_s == G + r_b_prev * B_s_prev + BL_s_prev - T - r_b_prev * B_cb_prev - dBL_s * p_bL
dH_s == H_s - H_s_prev
dH_s == dB_cb
dB_cb == B_cb - B_cb_prev
B_cb == B_s - B_h
BL_s == BL_h
ERr_bL == r_bL + chi * (p_bL_e - p_bL) / p_bL
r_bL == 1 / p_bL
p_bL_e == p_bL
CG_e == chi * (p_bL_e - p_bL) * BL_h
YD_r_e == YD_r_prev
r_b == r_b_cap
p_bL == p_bL_cap
dp_bL == p_bL - p_bL_prev
dBL_s == BL_s - BL_s_prev
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "theta": 0.1,
            "lambda_20": 0.1,
            "lambda_22": 0.2,
            "lambda_23": 0.1,
            "lambda_24": 0.2,
            "lambda_30": 0.3,
            "lambda_32": 0.01,
            "lambda_33": 0.5,
            "lambda_34": 0.1,
            "chi": 0.5,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.1 + x / 1e5,
                "postprocess": lambda x: x
            },
            "r_b_cap": {
                "transform": lambda x: jnp.clip(0.1 + x / 1e5, 0, 1.0),
                "postprocess": lambda x: jnp.clip(x, 0.0, 1)
           },
            "p_bL_cap": {
                "transform": lambda x: jnp.maximum(10 + x / 1e5, 1.0),
                "postprocess": lambda x: jnp.maximum(x, 1.0)
           }
        },
        "exclude_variables": "dB_s, dH_s, dB_cb, dBL_s, dp_bL, CG_e,",
    },

    "LP2": {
        "system": '''
Y == C + G
YD_r == Y - T + r_b_prev * B_h_prev + BL_h_prev
T == theta * (Y * r_b_prev * B_h_prev + BL_h_prev)
V == V_prev + YD_r - C + CG
CG == dp_bL * BL_h_prev
C == alpha_1 * YD_r_e + alpha_2 * V_prev
V_e == V_prev + YD_r_e - C + CG_e
H_h == V - B_h - p_bL * BL_h
H_d == V_e - B_d - p_bL * BL_d
B_d / V_e == lambda_20 + lambda_22 * r_b + lambda_23 * ERr_bL + lambda_24 * YD_r_e / V_e
BL_d * p_bL / V_e == lambda_30 + lambda_32 * r_b + lambda_33 * ERr_bL + lambda_34 * YD_r_e / V_e
B_h == B_d
BL_h == BL_d
dB_s == B_s - B_s_prev
dB_s == G + r_b_prev * B_s_prev + BL_s_prev - T - r_b_prev * B_cb_prev - dBL_s * p_bL
dH_s == H_s - H_s_prev
dH_s == dB_cb
dB_cb == B_cb - B_cb_prev
B_cb == B_s - B_h
BL_s == BL_h
ERr_bL == r_bL + chi * (p_bL_e - p_bL) / p_bL
r_bL == 1 / p_bL
CG_e == chi * (p_bL_e - p_bL) * BL_h
YD_r_e == YD_r_prev
r_b == r_b_cap
dp_bL == p_bL - p_bL_prev
dBL_s == BL_s - BL_s_prev
dp_bL_e == -beta_e * (p_bL_e_prev - p_bL) + add
dp_bL_e == p_bL_e - p_bL_e_prev
p_bL == (1 + z1 * beta_ - z2 * beta_) * p_bL_prev + add1
''',
        "params": {
            "alpha_1": 0.2,
            "alpha_2": 0.1,
            "theta": 0.1,
            "lambda_20": 0.1,
            "lambda_22": 0.2,
            "lambda_23": 0.1,
            "lambda_24": 0.2,
            "lambda_30": 0.3,
            "lambda_32": 0.01,
            "lambda_33": 0.5,
            "lambda_34": 0.1,
            "chi": 0.5,
            "beta_e": 0.005,
            "beta_": 0.005,
            "add_std": 10.0,
            "add1_std": 10.0,
            "top": 0.66,
            "bot": 0.33,
        },
        "actions": {
            "G": {
                "transform": lambda x: 0.1 + x / 1e5,
                "postprocess": lambda x: x
            },
            "r_b_cap": {
                "transform": lambda x: jnp.clip(0.1 + x / 1e5, 0, 1.0),
                "postprocess": lambda x: jnp.clip(x, 0.0, 1)
           },
        },
        "exclude_variables": "dB_s, dH_s, dB_cb, dBL_s, dp_bL, dp_bL_e",
        "stubs": {
            "add": "jax.random.normal(seed, (1,)) * add_std",
            "add1": "jax.random.normal(seed, (1,)) * add1_std",
            "TP": "BL_h_prev * p_bL_prev / (BL_h_prev * p_bL_prev + B_h_prev)",
            "z1": "(TP > top).astype(jnp.float32)",
            "z2": "(TP < bot).astype(jnp.float32)",
        },
        "initial_state_stub": lambda s: s.at[:,-4].add(1000.0),
    },

    "REG": {
        "system": '''
Y_S == C_S + G_S + X_S - IM_S
IM_S == mu_S * Y_S
X_S == IM_N
YD_S == Y_S - T_S + r_prev * B_S_h_prev
T_S == theta * (Y_S + r_prev * B_S_h_prev)
V_S == V_S_prev + (YD_S - C_S)
C_S == alpha_S_1 * YD_S + alpha_S_2 * V_S_prev
H_S_h == V_S - B_S_h
B_S_h / V_S == lambda_S_0 + lambda_S_1 * r - lambda_S_2 * YD_S / V_S
Y_N == C_N + G_N + X_N - IM_N
IM_N == mu_N * Y_N
X_N == IM_N
YD_N == Y_N - T_N + r_prev * B_N_h_prev
T_N == theta * (Y_N + r_prev * B_N_h_prev)
V_N == V_N_prev + (YD_N - C_N)
C_N == alpha_N_1 * YD_N + alpha_N_2 * V_N_prev
H_N_h == V_N - B_N_h
B_N_h / V_N == lambda_N_0 + lambda_N_1 * r - lambda_N_2 * YD_N / V_N
T == T_N + T_S
G == G_N + G_S
B_h == B_N_h + B_S_h
H_h == H_N_h + H_S_h
dB_s == B_s - B_s_prev
dB_s == G + r_prev * B_s_prev - (T - r_prev * B_cb_prev)
dH_s == H_s - H_s_prev
dH_s == dB_cb
dB_cb == B_cb - B_cb_prev
B_cb == B_s - B_h
r == r_cap
''',
        "params": {
            "mu_N": 0.3,
            "mu_S": 0.2,
            "alpha_N_1": 0.2,
            "alpha_N_2": 0.1,
            "alpha_S_1": 0.2,
            "alpha_S_2": 0.1,
            "lambda_N_0": 0.1,
            "lambda_N_1": 0.2,
            "lambda_N_2": 0.3,
            "lambda_S_0": 0.3,
            "lambda_S_1": 0.2,
            "lambda_S_2": 0.1,
        },
        "actions": {
            "G_N": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "G_S": {
                "transform": lambda x: 0.2 + x / 1e4,
                "postprocess": lambda x: x
            },
            "theta": {
                "transform": lambda x: jnp.clip(0.2 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            },
            "r_cap": {
                "transform": lambda x: jnp.clip(0.05 + x / 1e4, 0, 1),
                "postprocess": lambda x: jnp.clip(x, 0, 1)
            }
        },
        "exclude_variables": "dB_s, dH_s, dB_cb,"
    },
}

def process_system(system_info: dict) -> str:
    sym_sorted = lambda x: sorted(x, key=lambda s: s.name)

    eqs = parse_expr(', '.join(system_info["system"].split('\n')[1:-1]), evaluate=False)

    variables = set.union(*(eq.free_symbols for eq in eqs))
    param_vars = set(map(sympy.Symbol, system_info["params"].keys()))
    action_vars = set(map(sympy.Symbol, system_info["actions"].keys()))
    exclude_vars = set(parse_expr(system_info["exclude_variables"])) if len(system_info["exclude_variables"]) > 0 else set()
    prev_vars = set(filter(lambda v: "_prev" in v.name, variables))
    stub_vars = set(map(sympy.Symbol, system_info["stubs"])) if "stubs" in system_info else set()
    state_vars = variables.difference(param_vars | action_vars | prev_vars | stub_vars)

    sys = sympy.solve(eqs, state_vars, manual=True, warn=False, check=False, simplify=False)

    if type(sys) == list:
        assert len(sys) == 1, str(len(sys))
        sys = sys[0]

    sys = {sym: formula for sym, formula in sys.items() if sym not in exclude_vars}
    state_vars ^= exclude_vars

    keys = sys.keys()
    tmp, sys = sympy.cse(sys.values(), order='none')
    sys = zip(keys, sys)

    code = "@jit\ndef update(state: jnp.ndarray, action: jnp.ndarray, params: jnp.ndarray, seed: jnp.ndarray = None):\n"

    code += "\t" + ", ".join(map(lambda s: f"{str(s)}_prev", sym_sorted(state_vars))) + ", = state.T\n"
    code += "\t" + str(sym_sorted(action_vars))[1:-1] + ", = action.T\n"
    code += "\t" + str(sym_sorted(param_vars))[1:-1] + ", = params\n\n"

    if "stubs" in system_info:
        for stub in system_info["stubs"]:
            code += "\t" +stub + " = " + system_info["stubs"][stub] + "\n"

    for a in (tmp, sys):
        for l, r in a:
            code += "\t" + f"{pycode(l)} = {pycode(r)}\n"
        code += '\n'
    code += "\t" + f"return jnp.array([{str(sym_sorted(state_vars))[1:-1]}]).T\n"

    update_code = code

    local_namespace = {}
    exec(update_code, globals(), local_namespace)

    update = local_namespace["update"]

    return {
        "update_code": update_code,
        "update": update,
        "state_vars": str(sym_sorted(state_vars))[1:-1].split(', '),
        "action_vars": str(sym_sorted(action_vars))[1:-1].split(', '),
        "params": str(sym_sorted(param_vars))[1:-1].split(', '),
        "equations": eqs,
    }


def get_test_initial_states(state_dim):
    seed = jnp.array([623, 597], dtype="u4")
    n_states = 1024 * 16

def base_cost(s: jnp.ndarray, i: jnp.ndarray, t: jnp.ndarray):
    return ((s[..., i] - t) ** 2).sum(axis=-1)

# @partial(jit, static_argnames=["update", "action", "n_steps"])
def get_episode(update: Callable, system_params: jnp.ndarray, action: Callable, params: dict, initial_state: jnp.ndarray, n_steps: int, key: jnp.ndarray):
    state = initial_state
    states, actions = [initial_state], []

    for i in range(n_steps):
        key, seed = jax.random.split(key)

        act = action(params, state, seed)
        actions.append(act)

        state = update(state, act, system_params, seed)
        states.append(state)

    return jnp.stack(states, axis=1), jnp.stack(actions, axis=1)

# @partial(jit, static_argnames=["update", "action", "cost", "n_steps"])
def get_episode_cost(
    params: dict,
    update: Callable,
    system_params: jnp.ndarray,
    action: Callable,
    initial_state: jnp.ndarray,
    cost: Callable,
    n_steps: int,
    key: jnp.ndarray
):
    state = initial_state
    total_cost = cost(state)
    for i in range(n_steps):
        act = action(params, state)
        key, seed = jax.random.split(key)
        state = update(state, act, system_params, seed)
        total_cost += cost(state)
    return total_cost.mean() / n_steps


@partial(jit, static_argnames=["transforms"])
def transform_action(raw_action: jnp.ndarray, transforms: tuple) -> jnp.ndarray:
    assert raw_action.shape[-1] == len(transforms)
    return jnp.stack([transforms[i](raw_action[..., i]) for i in range(len(transforms))], axis=-1)

@jit
def action_logprob(mu: jnp.ndarray, action: jnp.ndarray, sigma: float) -> jnp.ndarray:
    dist = distrax.Normal(mu, sigma)
    return dist.log_prob(action).sum(axis=-1)

@partial(jit, static_argnames=["transforms"])
def sample_action(mu: jnp.ndarray, sigma: float, seed: jnp.ndarray, transforms: tuple) -> jnp.ndarray:
    dist = distrax.Normal(mu, sigma)
    act = jax.lax.stop_gradient(dist.sample(seed=seed))
    return transform_action(act, transforms)

@jit
def baseline_objective(x: jnp.ndarray, y: jnp.ndarray) -> jnp.float32:
    return ((x - y) ** 2).mean()

@jit
def policy_objective(r: jnp.ndarray, log_probs: jnp.ndarray) -> jnp.ndarray:
    return (r * log_probs).mean()

class MLP(nn.Module):
    dims: list[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.dims):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.dims) - 1:
                x = nn.relu(x)
        return x
