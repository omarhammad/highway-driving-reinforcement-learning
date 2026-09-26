# Highway Driving with Deep Reinforcement Learning

A simulation-based reinforcement learning project that trains agents to drive in `highway-env`. The repository explores Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), custom reward functions, denser traffic scenarios, saved checkpoints, and evaluation runs.

**Scope:** simulated highway driving. This project does not control a real vehicle.

## Project overview

The agent observes a highway scenario and selects driving actions through the Gymnasium-compatible `highway-env` environment. The experiments change traffic and road settings, shape rewards for driving behavior, train DQN or PPO policies with Stable-Baselines3, and inspect training logs and simulated episodes.

| Experiment | Focus |
| --- | --- |
| DQN | Training, checkpoints, custom rewards, and evaluation in modified highway scenarios. |
| PPO | Training and evaluation with a custom reward and configured action/observation spaces. |
| Environment comparison | Notebook experiments comparing reward or environment configurations. |

## Reward design and evaluation

The custom environment subclasses `HighwayEnvFast` and defines reward components for combinations of collision avoidance, speed, lane position, and, in one DQN variant, overtaking. Reward components are written to TensorBoard summaries. The training scripts save model checkpoints at regular intervals.

Evaluation scripts load saved policies and run simulated episodes. The PPO evaluation reports average episode reward and length over ten episodes. The repository includes training logs and checkpoints, but does not provide one standardized benchmark comparing every experiment.

## Repository structure

| Path | Contents |
| --- | --- |
| [`dqn_algo_hamed/`](dqn_algo_hamed/) | DQN training/evaluation scripts, custom reward, checkpoints, and logs. |
| [`dqn_algo_margarita/`](dqn_algo_margarita/) | DQN environment comparisons, training variations, evaluation, checkpoints, and logs. |
| [`ppo_algo/`](ppo_algo/) | PPO custom reward, training, evaluation, and related outputs. |

The folder names are preserved from the source repository.

## Technology

Python · Gymnasium · highway-env · Stable-Baselines3 · PyTorch · TensorBoard · TensorFlow summaries · Jupyter

## Running an experiment

1. Set up a Python environment with `gymnasium`, `highway-env`, `stable-baselines3`, `torch`, `tensorboard`, and `tensorflow`. Jupyter is needed for the notebooks.
2. Work from the relevant algorithm folder so the local custom environment module and relative paths resolve.
3. Review the environment configuration and checkpoint paths in its `Training.py` or `Evaluation.py` before running. Some paths are specific to the original development machine and must be changed.
4. Train a policy or point the evaluation script to an available checkpoint. Review logs with TensorBoard.

The archive does not include a pinned dependency file or a single reproducible entry point. These scripts document experiments and may require path and environment adjustments on another machine.
