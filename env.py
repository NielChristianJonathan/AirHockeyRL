import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AirHockeyEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.WIDTH, self.HEIGHT = 1920, 1080
        self.PADDLE_RADIUS = 50
        self.PUCK_RADIUS = 30
        self.GOAL_RADIUS = 200
        self.TABLE_MARGIN = 30
        self.PADDLE_MAX_SPEED = 20.0
        self.PUCK_MAX_SPEED = 30.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.opponent_policy = None
        self.time_on_side = 0
        self.reset()

    def set_opponent_policy(self, policy_fn):
        self.opponent_policy = policy_fn

    def reset(self, scorer=None, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.time_on_side = 0

        # Set player and AI positions
        self.player_pos = np.array([100.0, self.HEIGHT / 2], dtype=np.float32)
        self.ai_pos = np.array([self.WIDTH - 100.0, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.ai_vel = np.zeros(2, dtype=np.float32)

        # Determine puck starting position & velocity
        if scorer == "player":  # player scored → puck starts near AI
            self.puck_pos = np.array([self.WIDTH / 2 - 50, self.HEIGHT / 2], dtype=np.float32)
            self.puck_vel = np.zeros(2, dtype=np.float32)
        elif scorer == "ai":    # AI scored → puck starts near player
            self.puck_pos = np.array([self.WIDTH / 2 + 50, self.HEIGHT / 2], dtype=np.float32)
            self.puck_vel = np.zeros(2, dtype=np.float32)
        else:  # normal reset → center with random velocity
            self.puck_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            direction = 1 if np.random.rand() > 0.5 else -1
            speed = np.random.uniform(3, 8)
            self.puck_vel = np.array([direction * speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        pred_y = self.predict_puck_y_at_x(self.WIDTH * 0.75)

        return np.array([
            self.puck_pos[0] / self.WIDTH,
            self.puck_pos[1] / self.HEIGHT,
            self.puck_vel[0] / self.PUCK_MAX_SPEED,
            self.puck_vel[1] / self.PUCK_MAX_SPEED,
            self.player_pos[0] / self.WIDTH,
            self.player_pos[1] / self.HEIGHT,
            self.ai_pos[0] / self.WIDTH,
            self.ai_pos[1] / self.HEIGHT,
            pred_y / self.HEIGHT
        ], dtype=np.float32)

    def _move_ai(self, action):
        prev_ai = self.ai_pos.copy()
        self.ai_pos += np.array(action) * self.PADDLE_MAX_SPEED

        self.ai_pos[0] = np.clip(
            self.ai_pos[0],
            self.WIDTH/2 + self.PADDLE_RADIUS,
            self.WIDTH - self.TABLE_MARGIN - self.PADDLE_RADIUS
        )

        self.ai_pos[1] = np.clip(
            self.ai_pos[1],
            self.TABLE_MARGIN + self.PADDLE_RADIUS,
            self.HEIGHT - self.TABLE_MARGIN - self.PADDLE_RADIUS
        )

        self.ai_vel = self.ai_pos - prev_ai

    def _move_player(self, action):
        prev_player = self.player_pos.copy()
        self.player_pos += np.array(action) * self.PADDLE_MAX_SPEED

        self.player_pos[0] = np.clip(
            self.player_pos[0],
            self.TABLE_MARGIN + self.PADDLE_RADIUS,
            self.WIDTH/2 - self.PADDLE_RADIUS
        )

        self.player_pos[1] = np.clip(
            self.player_pos[1],
            self.TABLE_MARGIN + self.PADDLE_RADIUS,
            self.HEIGHT - self.TABLE_MARGIN - self.PADDLE_RADIUS
        )

        self.player_vel = self.player_pos - prev_player

    def _update_puck(self, dt=0.8):
        self.puck_pos += self.puck_vel * dt

        speed = np.linalg.norm(self.puck_vel)

        if speed < 2.0:
            self.puck_vel = self.puck_vel / (speed + 1e-6) * 2.0

        # pantulan atas/bawah
        if self.puck_pos[1] <= self.TABLE_MARGIN + self.PUCK_RADIUS:
            self.puck_pos[1] = self.TABLE_MARGIN + self.PUCK_RADIUS
            self.puck_vel[1] *= -1
        if self.puck_pos[1] >= self.HEIGHT - self.TABLE_MARGIN - self.PUCK_RADIUS:
            self.puck_pos[1] = self.HEIGHT - self.TABLE_MARGIN - self.PUCK_RADIUS
            self.puck_vel[1] *= -1

        # pantulan kiri/kanan non-goal
        goal_top = self.HEIGHT/2 - self.GOAL_RADIUS
        goal_bottom = self.HEIGHT/2 + self.GOAL_RADIUS
        if self.puck_pos[0] <= self.TABLE_MARGIN + self.PUCK_RADIUS and not (goal_top < self.puck_pos[1] < goal_bottom):
            self.puck_pos[0] = self.TABLE_MARGIN + self.PUCK_RADIUS
            self.puck_vel[0] *= -1
        if self.puck_pos[0] >= self.WIDTH - self.TABLE_MARGIN - self.PUCK_RADIUS and not (goal_top < self.puck_pos[1] < goal_bottom):
            self.puck_pos[0] = self.WIDTH - self.TABLE_MARGIN - self.PUCK_RADIUS
            self.puck_vel[0] *= -1

    def _handle_collisions(self):
        collision = False
        dist_ai = np.linalg.norm(self.puck_pos - self.ai_pos)
        dist_player = np.linalg.norm(self.puck_pos - self.player_pos)

        if dist_ai < (self.PADDLE_RADIUS + self.PUCK_RADIUS):
            collision = True
            self.resolve_collision(self.ai_pos, self.ai_vel)


        if dist_player < (self.PADDLE_RADIUS + self.PUCK_RADIUS):
            self.resolve_collision(self.player_pos, self.player_vel)

        return collision, dist_ai


    def step(self, ai_action, mode, player_action=None):
        self.steps += 1

        # gerakkan AI
        self._move_ai(ai_action)

        # gerakkan player/opponent
        if player_action is not None:
            if self.opponent_policy:
                player_action = self.opponent_policy(self._get_obs())
                self._move_player(player_action)
            else:
                player_action = np.random.uniform(-1, 1, size=2)
                self._move_player(player_action)

        # update puck & collisions
        self._update_puck()
        collision, dist_ai = self._handle_collisions()

        self._clamp_paddles()

        # check goal
        reward = 0.0
        done = False
        goal_top = self.HEIGHT/2 - self.GOAL_RADIUS
        goal_bottom = self.HEIGHT/2 + self.GOAL_RADIUS
        if self.puck_pos[0] <= self.TABLE_MARGIN + self.PUCK_RADIUS and goal_top < self.puck_pos[1] < goal_bottom:
            reward += 10.0
            done = True
        elif self.puck_pos[0] >= self.WIDTH - self.TABLE_MARGIN - self.PUCK_RADIUS and goal_top < self.puck_pos[1] < goal_bottom:
            reward -= 8.0
            done = True

        # reward shaping
        reward += self._compute_reward(dist_ai, collision, mode)

        # timestep limit
        if self.steps > 1000:
            done = True

        if self.puck_vel[0] < 0:
            self.time_on_side += 1
        else:
            self.time_on_side = 0

        self.last_info = {"reward": float(reward), "dist": float(dist_ai), "puck_x": float(self.puck_pos[0])}
        return self._get_obs(), reward, done, False, self.last_info


    def _clamp_paddles(self):
        self.player_pos[0] = np.clip(self.player_pos[0], self.TABLE_MARGIN, self.WIDTH / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN)

        self.ai_pos[0] = np.clip(self.ai_pos[0], self.WIDTH / 2, self.WIDTH - self.TABLE_MARGIN)
        self.ai_pos[1] = np.clip(self.ai_pos[1], self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN)

    def _compute_reward(self, dist, collision, mode):
        if mode == "attack":
            return self._compute_reward_defense(dist, collision)
        else:
            return self._compute_reward_attack(dist, collision)
    

    def _compute_reward_attack(self, dist, collision):
        r = 0.0

        puck_pos = self.puck_pos
        puck_vel = self.puck_vel
        agent_pos = self.ai_pos

        table_length = self.WIDTH

        # =========================================================
        # 2. Dense reward (Dreamer-style shaping)
        # =========================================================

        # (A) puck proximity
        dist = np.linalg.norm(puck_pos - agent_pos)
        max_distance = 800.0  # scaling pixel-based

        proximity_reward = 1 - np.clip(dist / max_distance, 0, 1)

        # (B) puck direction toward opponent goal (right side)
        goal_direction = np.array([-1.0, 0.0])

        puck_speed = np.linalg.norm(puck_vel)
        if puck_speed > 1e-6:
            puck_dir = puck_vel / puck_speed
        else:
            puck_dir = np.zeros(2)

        direction_reward = np.dot(puck_dir, goal_direction)

        # (C) puck velocity reward (strong hits encouraged)
        max_speed = self.PUCK_MAX_SPEED
        speed_reward = np.clip(puck_speed / max_speed, 0, 1)

        # (D) control reward (agent behind puck pushing forward)
        agent_to_puck = puck_pos - agent_pos
        norm = np.linalg.norm(agent_to_puck)

        if norm > 1e-6:
            control_dir = agent_to_puck / norm
        else:
            control_dir = np.zeros(2)

        control_reward = np.dot(control_dir, goal_direction)

        # =========================================================
        # 3. Collision bonus (your original idea)
        # =========================================================
        collision_bonus = 0.2 if collision else 0.0

        # =========================================================
        # 4. Combine (Dreamer-style weighting)
        # =========================================================
        r += 0.5 * proximity_reward
        r += 1.0 * direction_reward
        r += 0.3 * speed_reward
        r += 0.2 * control_reward
        r += collision_bonus

        return float(r)


    def _compute_reward_defense(self, dist, collision):
        reward = 0.0

        # =========================
        # 1. POSITIONING (PALING PENTING)
        # =========================
        goal_x = self.WIDTH
        goal_y = self.HEIGHT / 2

        puck = self.puck_pos
        agent = self.ai_pos

        # vector alignment (antara puck dan goal)
        goal_vec = puck - np.array([goal_x, goal_y])
        agent_vec = agent - np.array([goal_x, goal_y])

        cos_sim = np.dot(goal_vec, agent_vec) / (
            np.linalg.norm(goal_vec) * np.linalg.norm(agent_vec) + 1e-8
        )

        reward += 0.8 * cos_sim

        # =========================
        # 2. IKUTI JALUR PUCK
        # =========================
        pred_y = self.predict_puck_y_at_x(agent[0])
        y_dist = abs(pred_y - agent[1])

        reward += 1.0 * (1 - y_dist / self.HEIGHT)

        # =========================
        # 3. STAY DI DEFENSE LINE
        # =========================
        def_x = self.WIDTH * 0.8
        reward -= 0.5 * abs(agent[0] - def_x) / self.WIDTH

        # =========================
        # 4. INTERCEPT
        # =========================
        if collision:
            if self.puck_vel[0] < 0:
                reward += 2.0
            else:
                reward -= 1.0
        # =========================
        # 6. OVERCOMMIT PENALTY
        # =========================
        if agent[0] < self.WIDTH * 0.65:
            reward -= 0.5

        return reward

    def resolve_collision(self, paddle_pos, paddle_vel):
        normal = self.puck_pos - paddle_pos
        dist = np.linalg.norm(normal)

        if dist == 0:
            return

        normal = normal / dist

        rel_vel = self.puck_vel - paddle_vel
        approaching = np.dot(rel_vel, normal)

        # hanya kalau mendekat
        if approaching >= 0:
            return

        # reflect (correct physics)
        reflected = rel_vel - 2 * approaching * normal

        # === TAMBAHAN DI SINI ===
        paddle_speed = np.linalg.norm(paddle_vel)

        # faktor pengaruh kecepatan paddle
        boost_factor = 0.3

        boost = boost_factor * paddle_speed * normal

        # velocity baru
        self.puck_vel = 0.98 * (reflected + paddle_vel) + boost

        # limit speed
        speed = np.linalg.norm(self.puck_vel)
        if speed > self.PUCK_MAX_SPEED:
            self.puck_vel = self.puck_vel / speed * self.PUCK_MAX_SPEED

        # anti-sticking (strong)
        self.puck_pos = paddle_pos + normal * (self.PADDLE_RADIUS + self.PUCK_RADIUS + 0.5)

    def predict_puck_y_at_x(self, target_x):
        vx, vy = self.puck_vel

        if abs(vx) < 1e-5:
            return self.puck_pos[1]

        t = (target_x - self.puck_pos[0]) / vx
        if t < 0:
            return self.puck_pos[1]
        y = self.puck_pos[1] + vy * t

        top = self.TABLE_MARGIN
        bottom = self.HEIGHT - self.TABLE_MARGIN
        H = bottom - top

        y_rel = y - top
        y_mod = y_rel % (2 * H)

        y_final = y_mod if y_mod < H else 2 * H - y_mod
        return y_final + top