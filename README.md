# Korean 4-Ball Billiards: A Continuous, Deterministic, Sparse-Reward Benchmark Solved by Inference-Time Search

Code accompanying the CS377 final project (Team 5).
**Doyeol Oh · Byungmo Kang · Seojun Park** — KAIST

| | |
|---|---|
| 📄 **Paper** | [`docs/paper.pdf`](docs/paper.pdf) |
| 🖼️ **Slides** | [`docs/slides.pdf`](docs/slides.pdf) |
| 🎥 **Talk** | https://youtu.be/Z-ZltX3-1vo |

---

## TL;DR

Korean 4-ball (*sagu*) is a pocketless carom game: you score one point when your cue ball
caroms off **both** red balls in a single shot. We cast it as a **continuous, deterministic,
sparse-reward** RL problem with a fast, exact NumPy physics simulator, and ask what actually
breaks the score ceiling.

- **Learning supplies the prior.** Off-policy SAC beats PPO ~2.5× on the plain sparse task;
  random-start + continue-on-miss generalizes best — but the bare task plateaus below 1 point/inning.
- **Structure-free aids don't help.** A staged action curriculum and a learned reward-model bonus
  (trained on 10⁷ offline shots) both fail to break the ceiling.
- **Geometry makes it competent.** A first-contact aim constraint plus four carom features lift SAC
  from **0.487 → 6.460** points/inning.
- **Search scales it.** Greedy depth-2 lookahead — the simulator as its own verifier, no learned value —
  turns a ~6-point policy into chains of up to **8,451 consecutive scoring shots at 99.9% per-shot success**.

> The arc is a small re-enactment of the bitter lesson: when an exact, fast forward model is
> available, the highest-leverage move is not a cleverer reward but spending compute at inference time.

## Repository layout

```
billiards/                  Core simulator + Gym environments
  physics/                  Event-driven shot simulator
    state.py                Ball/table/action data structures
    cue_impact.py           Marlow instantaneous-point cue model
    dynamics.py             Free flight: slip/roll friction, spin decay
    collisions.py           Ball–cushion (Han 2005) + ball–ball restitution
    simulator.py            simulate_shot(): one shot → trajectory, events, score
  env.py                    Billiards4BallEnv — single-shot env
  inning_env.py             Billiards4BallInningEnv — multi-shot inning;
                            aim constraint, extra features, reward shaping live here
  wrappers/
    random_start_env.py     Randomised mid-rack starting layouts
    scoring_potential_rm_env.py   Dense reward-model bonus V_rm(s')   (§3.2)
    vsa_rm_env.py           State-action reward-model bonus variant   (§3.2)
    action_stage_env.py     Staged action-space curriculum            (§3.1)
    curriculum_start_env.py Start-state curriculum control            (§3.1)

experiments/                Training, evaluation, analysis drivers
  run_inning_sac.py         SAC / PPO / TD3 on the plain task         (§2.1, §2.2)
  run_inning_random.py      Random-start training + cross-eval        (§2.2)
  run_inning_matrix.py      Algorithm × seed sweep                    (§2.2)
  run_inning_curriculum.py  Start-curriculum control                  (§3.1)
  run_inning_action_curriculum.py  Staged action curriculum (failed)  (§3.1)
  build_v_sa_rm.py          Train V_rm from 10⁷ random offline shots  (§3.2)
  run_inning_sacrm.py       SAC + reward-model bonus (main 3.2 run)    (§3.2)
  run_inning_sacrm_sa.py    SAC + state-action RM bonus               (§3.2)
  eval_policy.py            Unified canonical + random evaluation
  summarize_domain_knowledge.py    Domain-knowledge ablation tables   (§4)
  configs.py                Shared sweep grids (seeds, alpha)

exp_4_lookahead/
  search_multiseed_h2.py    Greedy depth-2 lookahead search           (§5)

tests/                      Pytest suite for the simulator and envs
docs/                       Paper, slides, and engineering write-ups
```

`docs/PROJECT_OVERVIEW.md` and `docs/MODEL.md` are the long-form engineering references for the
simulator and physics models.

## Paper → code map

| Paper section | Code |
|---|---|
| §1 Domain & environment | `billiards/physics/`, `billiards/inning_env.py` |
| §2.1 SAC vs PPO vs TD3 | `experiments/run_inning_sac.py`, `run_inning_matrix.py` |
| §2.2 Four training paradigms | `run_inning_sac.py`, `run_inning_random.py`, `eval_policy.py` |
| §3.1 Staged action curriculum (failed) | `run_inning_action_curriculum.py`, `wrappers/action_stage_env.py` |
| §3.2 Learned reward-model guidance | `build_v_sa_rm.py`, `run_inning_sacrm.py`, `wrappers/scoring_potential_rm_env.py`, `wrappers/vsa_rm_env.py` |
| §4.1 Aim constraint | `inning_env.py` (`_apply_aim_constraint`, `constrain_aim=True`) |
| §4.2 Extra geometric features | `inning_env.py` (`_obs`, `extra_features=True`) |
| §4.3 Reward shaping | `inning_env.py` (gentle / near-miss / foul-penalty flags) |
| §5 Inference-time lookahead search | `exp_4_lookahead/search_multiseed_h2.py` |

## Setup

```bash
# Python ≥ 3.11. Using uv (recommended):
uv sync
# or with pip:
pip install -e .
```

## Quickstart

```bash
# §2 Plain sparse task — SAC baseline (random-start + continue-on-miss)
python -m experiments.run_inning_sac --algo sac --total_steps 400000 --seed 0 \
    --random_start --continue_on_miss

# §4 Domain knowledge — aim constraint + extra geometric features (the 6.46-pt run)
python -m experiments.run_inning_sac --algo sac --total_steps 1000000 --seed 0 \
    --random_start --continue_on_miss --constrain_aim --extra_features

# §3.2 Learned reward-model bonus — first build V_rm, then train with it
python -m experiments.build_v_sa_rm --n_states 5000 --n_actions 100 --out_dir runs/rm
python -m experiments.run_inning_sacrm --seed 0 --rm_path runs/rm/<checkpoint>.pt

# §5 Inference-time depth-2 lookahead over a trained 3-seed ensemble
python -m exp_4_lookahead.search_multiseed_h2 \
    --policies seed0/policy.zip seed1/policy.zip seed2/policy.zip \
    --k1_per_policy 50 --n_episodes 10 --out_dir runs/lookahead

# Tests
pytest
```

Run any script with `--help` for its full flag set (every flag in the paper is exposed —
shaping terms, aim-window width, search budget, etc.). Training and search write run artifacts
(checkpoints, parquet rollouts, CSV curves) into local output directories that are git-ignored.

## Environment at a glance

- **Observation** — 28-D (4 balls × 7 state features); 32-D with `extra_features=True`.
- **Action** — 4-D continuous: aim angle θ, relative power p, side/vertical cue-tip offsets a, b.
- **Reward** — the rule-defined carom score {0, 1} per shot; no shaping unless requested.
- **Episode** — an *inning* of up to `max_shots` shots, terminated by the first miss or foul.

One shot simulates in ≈ 5 ms on a single CPU core, so the forward model is cheap enough to use
as the verifier inside inference-time search.
