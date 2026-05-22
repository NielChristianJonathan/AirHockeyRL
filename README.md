# AirHockeyRL

A 2D Air Hockey game where you play against an AI agent trained using **Reinforcement Learning** (PPO, TD3) with optional **RLHF** (Reinforcement Learning from Human Feedback).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green.svg)
![Stable Baselines3](https://img.shields.io/badge/StableBaselines3-latest-orange.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-latest-yellow.svg)

---

## Gameplay

- Control your paddle using the **mouse**
- You play on the **left side**, the AI plays on the **right side**
- First to score the most goals in **3 minutes** wins
- The AI dynamically switches between **attack** and **defense** modes

---

## Project Structure

```
AirHockeyRL/
│
├── training/                        # Training notebooks
│   ├── Train_ppo_attack.ipynb
│   ├── Train_ppo_defense.ipynb
│   ├── Train_rlhf_attack_defense.ipynb
│   ├── Train_td3_attack.ipynb
│   └── Train_td3_defense.ipynb
│
├── game.py                          # Main game — run this to play
├── env.py                           # AirHockeyEnv (Gymnasium environment)
├── RLHF_input_nilai1game.py         # Collect human preference scores
├── RLHF_input_perbandingan2game.py  # Compare two games for RLHF
├── game_preference.py               # Compare model with/without RLHF
│
├── attack_ppo_airhockey.zip         # Trained PPO attack model
├── defense_ppo_airhockey.zip        # Trained PPO defense model
├── attack_td3_airhockey.zip         # Trained TD3 attack model
├── defense_td3_airhockey.zip        # Trained TD3 defense model
├── attack_rlhf.zip                  # Trained RLHF + PPO attack model
├── defense_rlhf.zip                 # Trained RLHF + PPO defense model
│
├── feedback_data.jsonl              # Human feedback data (RLHF)
├── preference_attack.jsonl          # Preference data for attack
├── preference_defense.jsonl         # Preference data for defense
├── rlhf_stats.json                  # RLHF training statistics
│
├── ball.png                         # Puck image asset
├── blue.png                         # Player paddle image
├── red.png                          # AI paddle image
│
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/NielChristianJonathan/AirHockeyRL.git
cd AirHockeyRL
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Play the game

```bash
python game.py
```

---

## AI Architecture

The AI agent uses a **dual-model strategy** that dynamically switches between two modes:

| Mode | Algorithm | Trigger |
|------|-----------|---------|
| **Attack** | PPO | Puck is in AI's half and moving toward player's goal |
| **Defense** | PPO | Puck is behind the AI or moving toward AI's goal |

The switch logic uses puck position (`puck_x`) and puck velocity (`puck_vel`) with a cooldown to prevent rapid switching.

---

## Training

All training is done inside Jupyter Notebooks in the `training/` folder.

| Notebook | Algorithm | Role |
|----------|-----------|------|
| `Train_ppo_attack.ipynb` | PPO | Attack agent |
| `Train_ppo_defense.ipynb` | PPO | Defense agent |
| `Train_td3_attack.ipynb` | TD3 | Attack agent |
| `Train_td3_defense.ipynb` | TD3 | Defense agent |
| `Train_rlhf_attack_defense.ipynb` | RLHF + PPO | Attack & Defense with human feedback |

### Environment: `AirHockeyEnv`

Built on **Gymnasium**, the environment features:

- **Observation space** (9 values): puck position & velocity, player position, AI position, predicted puck Y position
- **Action space**: continuous 2D movement `[-1, 1]`
- **Reward shaping**: separate reward functions for attack and defense roles
- **Physics**: realistic puck bouncing, collision resolution with velocity transfer

---

## RLHF Pipeline

Human feedback is collected to improve the reward model:

1. **`RLHF_input_nilai1game.py`** — Human watches a game and gives a score
2. **`RLHF_input_perbandingan2game.py`** — Human compares two gameplay clips and picks the better one
3. **`game_preference.py`** — Side-by-side comparison between standard PPO and RLHF-enhanced model
4. Feedback is saved to `feedback_data.jsonl` and used to retrain the reward model

---

## Requirements

```
pygame
stable-baselines3
gymnasium
numpy
```

Install all at once:

```bash
pip install pygame stable-baselines3 gymnasium numpy
```

---

## Controls

| Input | Action |
|-------|--------|
| Mouse movement | Move your paddle |
| `ESC` | Pause / Unpause |
| `R` | Restart game |
| `Q` | Quit |

---

## Author

**NielChristianJonathan & FeliciaCalista**
[github.com/NielChristianJonathan](https://github.com/NielChristianJonathan) 
[github.com/FeliciaCalista](https://github.com/FeliciaCalista) 
