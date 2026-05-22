import pygame
import sys
import numpy as np
import numpy
import math
import json
import os
import random
from datetime import datetime
from stable_baselines3 import PPO
from env import AirHockeyEnv
import time

STATS_FILE = "rlhf_stats.json"

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
YELLOW = (255, 255, 0)
FPS = 60

sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric

class AirHockeyGame:
    def __init__(self):

        pygame.init()

        # ENV (AI WORLD)
        self.env = AirHockeyEnv()
        self.obs, _ = self.env.reset()

        self.WIDTH = self.env.WIDTH
        self.HEIGHT = self.env.HEIGHT
        self.TABLE_MARGIN = self.env.TABLE_MARGIN
        self.GOAL_RADIUS = self.env.GOAL_RADIUS
        self.PADDLE_RADIUS = self.env.PADDLE_RADIUS
        self.switch_cooldown = 0
        self.mode = "attack"

        self.player_action = np.array([0.0, 0.0])

        # SCREEN (FULLSCREEN)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()

        pygame.display.set_caption("Air Hockey Game")

        # SCALING (VISUAL ONLY)
        self.scale = min(
            self.screen_width / self.WIDTH,
            self.screen_height / self.HEIGHT
        )

        self.table_width = int(self.WIDTH * self.scale)
        self.table_height = int(self.HEIGHT * self.scale)

        self.offset_x = (self.screen_width - self.table_width) // 2
        self.offset_y = (self.screen_height - self.table_height) // 2

        # MODEL
        # Model SEBELUM RLHF
        self.model_defense_before = PPO.load(
            "defense_ppo_airhockey.zip",
            device="cpu"
        )
        self.model_attack_before = PPO.load(
            "attack_ppo_airhockey.zip",
            device="cpu"
        )
        # Model SESUDAH RLHF
        self.model_defense_after = PPO.load(
            "defense_rlhf.zip",
            device="cpu"
        )
        self.model_attack_after = PPO.load(
            "attack_rlhf.zip",
            device="cpu"
        )

        # Aktif model (akan di-set saat mulai round)
        self.model_defense = self.model_defense_before
        self.model_attack = self.model_attack_before
        
        # TIME / UI
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Calibri", 30)
        self.game_duration = 5
        self.start_ticks = pygame.time.get_ticks()
        self.time_left = self.game_duration


        # BACKGROUND
        self.hex_bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.hex_bg.fill(BLACK)
        self.draw_hex_grid(self.hex_bg, 35)
        self.hex_bg_scaled = pygame.transform.smoothscale(
            self.hex_bg,
            (self.table_width, self.table_height)
        )

        # ASSETS
        self.player_img = pygame.image.load("blue.png").convert_alpha()
        self.enemy_img = pygame.image.load("red.png").convert_alpha()
        self.puck_img = pygame.image.load("ball.png").convert_alpha()

        size_paddle = int(self.env.PADDLE_RADIUS * 2 * self.scale)
        size_puck = int(self.env.PUCK_RADIUS * 2 * self.scale)

        self.player_img = pygame.transform.smoothscale(self.player_img, (size_paddle, size_paddle))
        self.enemy_img = pygame.transform.smoothscale(self.enemy_img, (size_paddle, size_paddle))
        self.puck_img = pygame.transform.smoothscale(self.puck_img, (size_puck, size_puck))

        # ENV RESET (WAJIB DARI ENV)
        self.obs, _ = self.env.reset()

        # UI STATE (GAME ONLY)
        self.score_player = 0
        self.score_ai = 0
        
        self.menu_options = ["Resume", "Restart", "Exit"]
        self.menu_index = 0

        self.state = "menu"  
        self.countdown_start = None
        self.countdown_value = 3

        self.menu_options_main = ["Play", "Exit"]
        self.menu_index_main = 0
        self.menu_buttons = []

        # RLHF EXPERIMENT
        # "before" = sebelum RLHF, "after" = sesudah RLHF
        self.model_order = ["before", "after"]
        random.shuffle(self.model_order)

        self.game_round = 1          # round saat ini (1 atau 2)
        self.round_scores = []       # [(score_player, score_ai), ...] per round
        self.vote_buttons = []       # tombol di layar voting
        self.voted_model = None

        # STATS (dari file JSON)
        self.stats = self.load_stats()



    # STATS: load / save / record

    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        # Default struktur kosong
        return {
            "total_sessions": 0,
            "votes": [],
            "summary": {
                "before_rlhf_votes": 0,
                "after_rlhf_votes": 0,
                "before_rlhf_pct": 0.0,
                "after_rlhf_pct": 0.0
            }
        }

    def save_stats(self):
        try:
            with open(STATS_FILE, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"[WARN] Gagal simpan stats: {e}")

    def record_vote(self, voted_label):
        """Catat hasil vote sesi ini. voted_label = 'Model A' atau 'Model B'."""
        # Tentukan tag aktual (before/after) dari pilihan
        idx = 0 if voted_label == "Model A" else 1
        voted_tag = self.model_order[idx]

        session = {
            "session": self.stats["total_sessions"] + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_order": list(self.model_order),
            "scores": [list(s) for s in self.round_scores[:2]],
            "voted": voted_label,
            "voted_tag": voted_tag
        }

        self.stats["votes"].append(session)
        self.stats["total_sessions"] += 1

        # Hitung ulang summary
        before = sum(1 for v in self.stats["votes"] if v["voted_tag"] == "before")
        after  = sum(1 for v in self.stats["votes"] if v["voted_tag"] == "after")
        total  = before + after
        self.stats["summary"] = {
            "before_rlhf_votes": before,
            "after_rlhf_votes":  after,
            "before_rlhf_pct":   round(before / total * 100, 1) if total else 0.0,
            "after_rlhf_pct":    round(after  / total * 100, 1) if total else 0.0
        }

        self.save_stats()

    def run(self):
        pygame.mouse.set_visible(False)
        running = True

        while running:

            self.handle_input()

            if self.state in ["menu", "paused", "game_over", "vote"]:
                pygame.mouse.set_visible(True)
            else:
                pygame.mouse.set_visible(False)

            if self.state == "countdown":
                    elapsed = (pygame.time.get_ticks() - self.countdown_start) // 1000
                    self.countdown_value = 3 - elapsed

                    if self.countdown_value <= 0:
                        self.state = "playing"
                        self.start_ticks = pygame.time.get_ticks()

            if self.state == "playing":

                ai_action = self.ai_model_control(self.obs)
                self.player_action = self.get_player_action()

                obs, reward, done, _, _ = self.env.step(ai_action, None)
                self.obs = obs

                if done:
                    if self.env.puck_pos[0] < self.env.TABLE_MARGIN:
                        self.score_ai += 1
                        self.env.reset("ai")
                    elif self.env.puck_pos[0] > self.env.WIDTH - self.env.TABLE_MARGIN:
                        self.score_player += 1
                        self.env.reset("player")

                    self.obs = self.env._get_obs()

                self.update_timer()

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def ai_model_control(self, obs):
        puck_x = obs[0]
        puck_vel = obs[2]
        ai_x = obs[6]

        # PRIORITY: bola di belakang AI
        if puck_x > ai_x + 0.02:
            self.mode = "defense"

        # SWITCH LOGIC
        elif self.switch_cooldown == 0:
            if self.mode == "defense":
                if puck_vel == 0:
                    self.mode = "attack"
                    self.switch_cooldown = 15
                if puck_x > 0.55 and puck_vel >= -0.2:
                    self.mode = "attack"
                    self.switch_cooldown = 15

            elif self.mode == "attack":
                if puck_x < 0.45 or puck_vel < -0.3:
                    self.mode = "defense"
                    self.switch_cooldown = 15

        # cooldown reduce
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1

        # ACTION
        if self.mode == "attack":
            action, _ = self.model_attack.predict(obs, deterministic=True)
        else:
            action, _ = self.model_defense.predict(obs, deterministic=True)

        return action

    def get_player_action(self):
        return self.player_action

    def to_screen(self, pos):
        x = pos[0] * self.table_width / self.env.WIDTH + self.offset_x
        y = pos[1] * self.table_height / self.env.HEIGHT + self.offset_y
        return int(x), int(y)
    
    def to_world(self, pos):
        x = (pos[0] - self.offset_x) * self.WIDTH / self.table_width
        y = (pos[1] - self.offset_y) * self.HEIGHT / self.table_height
        return np.array([x, y])
    
    def update_timer(self):
        if self.state == "paused":
            return

        seconds_passed = (pygame.time.get_ticks() - self.start_ticks) // 1000
        self.time_left = max(0, self.game_duration - seconds_passed)

        if self.time_left == 0:
            # Simpan skor round ini (hanya sekali)
            if len(self.round_scores) < self.game_round:
                self.round_scores.append((self.score_player, self.score_ai))
            if self.game_round < 2:
                self.state = "game_over"
            else:
                self.state = "game_over"


    def _handle_menu_action(self):
        choice = self.menu_options[self.menu_index]

        if choice == "Resume":
            self.state = "playing"

        elif choice == "Restart":
            self.restart_game()

        elif choice == "Exit":
            pygame.quit()
            sys.exit()


    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                mouse_pos = pygame.mouse.get_pos()

                # MENU
                if self.state == "menu":
                    for rect, option in self.menu_buttons:
                        if rect.collidepoint(mouse_pos):
                            if option == "Play":
                                self.round_scores.clear()
                                self.game_round = 1
                                self.load_model_for_round(self.game_round)
                                self.score_player = 0
                                self.score_ai = 0
                                self.state = "countdown"
                                self.countdown_start = pygame.time.get_ticks()

                            elif option == "Exit":
                                pygame.quit()
                                sys.exit()
                
                if self.state == "paused":
                    for rect, option in self.pause_buttons:
                        if rect.collidepoint(mouse_pos):
                            if option == "Resume":
                                self.state = "playing"

                            elif option == "Restart":
                                self.restart_game()

                            elif option == "Exit":
                                self.state = "menu"
                
                if self.state == "game_over":
                    for rect, option in self.gameover_buttons:
                        if rect.collidepoint(mouse_pos):
                            if option == "Restart":
                                self.round_scores.clear()
                                self.game_round = 1
                                import random
                                self.model_order = ["before", "after"]
                                random.shuffle(self.model_order)
                                self.load_model_for_round(self.game_round)
                                self.restart_game()

                            elif option == "Next Round":
                                self.restart_game(new_round=True)

                            elif option == "Vote":
                                self.state = "vote"

                            elif option == "Menu":
                                self.state = "menu"

                if self.state == "vote":
                    for rect, option in self.vote_buttons:
                        if rect.collidepoint(mouse_pos):
                            # option = "Model A" atau "Model B"
                            # Simpan pilihan (opsional: bisa di-print atau di-log)
                            self.record_vote(option) 
                            self.voted_model = option
                            self.state = "menu"
                            # Reset untuk sesi berikutnya
                            self.round_scores.clear()
                            self.game_round = 1
                            import random
                            self.model_order = ["before", "after"]
                            random.shuffle(self.model_order)
                            self.load_model_for_round(self.game_round)

            if event.type == pygame.KEYDOWN:
            
                if self.state == "game_over":
                    if event.key == pygame.K_r:
                        self.round_scores.clear()
                        self.game_round = 1
                        import random
                        self.model_order = ["before", "after"]
                        random.shuffle(self.model_order)
                        self.load_model_for_round(self.game_round)
                        self.restart_game()

                    elif event.key == pygame.K_ESCAPE:
                        self.state = "menu"

                if self.state == "paused":
                    if event.key == pygame.K_ESCAPE:
                        self.state = "playing"

                    elif event.key == pygame.K_UP:
                        self.menu_index = (self.menu_index - 1) % len(self.menu_options)

                    elif event.key == pygame.K_DOWN:
                        self.menu_index = (self.menu_index + 1) % len(self.menu_options)

                    elif event.key == pygame.K_RETURN:
                        self._handle_menu_action()

                    # shortcut
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self.state = "playing"

                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

                # MAIN MENU
                if self.state == "menu":

                    if event.key == pygame.K_UP:
                        self.menu_index_main = (self.menu_index_main - 1) % len(self.menu_options_main)

                    elif event.key == pygame.K_DOWN:
                        self.menu_index_main = (self.menu_index_main + 1) % len(self.menu_options_main)

                    elif event.key == pygame.K_RETURN:
                        choice = self.menu_options_main[self.menu_index_main]

                        if choice == "Play":
                            self.round_scores.clear()
                            self.game_round = 1
                            self.load_model_for_round(self.game_round)
                            self.score_player = 0
                            self.score_ai = 0
                            self.state = "countdown"
                            self.countdown_start = pygame.time.get_ticks()

                        elif choice == "Exit":
                            pygame.quit()
                            sys.exit()

                # GAME MODE
                else:
                    if event.key == pygame.K_ESCAPE:
                        self.state = "paused"
                        self.menu_index = 0

                    elif event.key == pygame.K_r:
                        self.env.reset()

            
        # MOUSE → WORLD POSITION
        if self.state == "playing":
            mouse_pos = pygame.mouse.get_pos()
            world_pos = self.to_world(mouse_pos)

            world_pos[0] = np.clip(
                world_pos[0],
                self.TABLE_MARGIN + self.PADDLE_RADIUS,
                self.WIDTH / 2 - self.PADDLE_RADIUS
            )

            world_pos[1] = np.clip(
                world_pos[1],
                self.TABLE_MARGIN + self.PADDLE_RADIUS,
                self.HEIGHT - self.TABLE_MARGIN - self.PADDLE_RADIUS
            )

            prev = self.env.player_pos.copy()
            self.env.player_pos = world_pos
            self.env.player_vel = world_pos - prev

    def load_model_for_round(self, round_num):
        """Set model aktif sesuai urutan round (player tidak tahu yang mana)."""
        tag = self.model_order[round_num - 1]
        if tag == "before":
            self.model_defense = self.model_defense_before
            self.model_attack = self.model_attack_before
        else:
            self.model_defense = self.model_defense_after
            self.model_attack = self.model_attack_after

    def restart_game(self, new_round=False):
        self.env.reset()
        self.obs = self.env._get_obs()
        self.mode = "attack"

        self.score_player = 0
        self.score_ai = 0

        self.start_ticks = pygame.time.get_ticks()
        self.time_left = self.game_duration

        if new_round:
            self.game_round += 1
            self.load_model_for_round(self.game_round)

        self.state = "countdown"
        self.countdown_start = pygame.time.get_ticks()


    def draw_main_menu(self, screen):
        width, height = screen.get_size()

        # BACKGROUND
        self.draw_gradient_bg(screen, (10, 10, 30), (30, 10, 50))

        # TITLE
        font_title = pygame.font.SysFont("Calibri", 90, bold=True)

        title = font_title.render("AIR HOCKEY", True, (255, 255, 255))
        glow = font_title.render("AIR HOCKEY", True, (0, 200, 255))
        glow.set_alpha(80)

        screen.blit(glow, glow.get_rect(center=(width//2, height//3 + 5)))
        screen.blit(title, title.get_rect(center=(width//2, height//3)))

        # BUTTONS
        font_menu = pygame.font.SysFont("Calibri", 50)

        mouse_pos = pygame.mouse.get_pos()
        self.menu_buttons = []

        for i, option in enumerate(self.menu_options_main):

            rect = pygame.Rect(width//2 - 180, height//2 + i*90 - 35, 360, 70)

            # hover effect
            if rect.collidepoint(mouse_pos):
                color = (255, 215, 0)
                scale = 1.1
            else:
                color = (200, 200, 200)
                scale = 1.0

            # button glow border
            pygame.draw.rect(screen, (0, 200, 255), rect, 2, border_radius=15)

            # text render
            text = font_menu.render(option, True, color)

            # scale effect
            new_w = int(text.get_width() * scale)
            new_h = int(text.get_height() * scale)
            text = pygame.transform.smoothscale(text, (new_w, new_h))

            screen.blit(text, text.get_rect(center=rect.center))

            self.menu_buttons.append((rect, option))


    def draw_game_over(self, screen):
        width, height = screen.get_size()

        # OVERLAY GELAP
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        # PANEL
        panel_w = int(width * 0.4)
        panel_h = int(height * 0.55)

        panel_rect = pygame.Rect(
            width // 2 - panel_w // 2,
            height // 2 - panel_h // 2,
            panel_w,
            panel_h
        )

        pygame.draw.rect(screen, (30, 30, 30), panel_rect, border_radius=20)
        pygame.draw.rect(screen, (255, 215, 0), panel_rect, 3, border_radius=20)

        # FONT
        font_title = pygame.font.SysFont("Calibri", 70, bold=True)
        font_round = pygame.font.SysFont("Calibri", 35)
        font_winner = pygame.font.SysFont("Calibri", 45)
        font_score = pygame.font.SysFont("Calibri", 40)
        font_button = pygame.font.SysFont("Calibri", 35)

        # TITLE
        title = font_title.render("GAME OVER", True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(width//2, panel_rect.top + 70)))

        # ROUND LABEL
        round_label = font_round.render(f"Round {self.game_round} of 2  —  Model {'A' if self.game_round == 1 else 'B'}", True, (180, 180, 255))
        screen.blit(round_label, round_label.get_rect(center=(width//2, panel_rect.top + 120)))

        # WINNER
        if self.score_player > self.score_ai:
            winner = "PLAYER WINS"
        elif self.score_ai > self.score_player:
            winner = "AI WINS"
        else:
            winner = "DRAW"

        win_text = font_winner.render(winner, True, (255, 215, 0))
        screen.blit(win_text, win_text.get_rect(center=(width//2, panel_rect.top + 165)))

        # SCORE
        score_text = f"{self.score_player}  -  {self.score_ai}"
        score_render = font_score.render(score_text, True, (200, 200, 200))
        screen.blit(score_render, score_render.get_rect(center=(width//2, panel_rect.top + 215)))

        # BUTTONS (beda per round)
        mouse_pos = pygame.mouse.get_pos()
        self.gameover_buttons = []

        if self.game_round < 2:
            # Round 1: lanjut ke round berikutnya atau restart dari awal
            buttons = [
                ("Next Round", panel_rect.top + 295),
                ("Restart", panel_rect.top + 360),
                ("Menu", panel_rect.top + 425)
            ]
        else:
            # Round 2: vote atau restart dari awal
            buttons = [
                ("Vote", panel_rect.top + 295),
                ("Restart", panel_rect.top + 360),
                ("Menu", panel_rect.top + 425)
            ]

        for text, y in buttons:
            rect = pygame.Rect(width//2 - 150, y - 25, 300, 50)

            if rect.collidepoint(mouse_pos):
                color = (255, 215, 0)
                scale = 1.1
            else:
                color = (180, 180, 180)
                scale = 1.0

            render = font_button.render(text, True, color)
            new_w = int(render.get_width() * scale)
            new_h = int(render.get_height() * scale)
            render = pygame.transform.smoothscale(render, (new_w, new_h))

            screen.blit(render, render.get_rect(center=rect.center))
            self.gameover_buttons.append((rect, text))


    def draw_vote_screen(self, screen):
        width, height = screen.get_size()

        # Background
        self.draw_gradient_bg(screen, (10, 10, 30), (30, 10, 50))

        # Panel
        panel_w = int(width * 0.55)
        panel_h = int(height * 0.75)
        panel_rect = pygame.Rect(width//2 - panel_w//2, height//2 - panel_h//2, panel_w, panel_h)

        pygame.draw.rect(screen, (20, 20, 40), panel_rect, border_radius=20)
        pygame.draw.rect(screen, (0, 200, 255), panel_rect, 3, border_radius=20)

        font_title = pygame.font.SysFont("Calibri", 55, bold=True)
        font_sub   = pygame.font.SysFont("Calibri", 30)
        font_score = pygame.font.SysFont("Calibri", 32)
        font_btn   = pygame.font.SysFont("Calibri", 38, bold=True)
        font_note  = pygame.font.SysFont("Calibri", 26)

        # Title
        title = font_title.render("Mana Model yang Lebih Baik?", True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(width//2, panel_rect.top + 55)))

        sub = font_sub.render("Kamu baru saja melawan dua model AI yang berbeda.", True, (160, 160, 200))
        screen.blit(sub, sub.get_rect(center=(width//2, panel_rect.top + 100)))

        # Score summary (round 1 & 2)
        y_score = panel_rect.top + 150
        labels = ["Model A", "Model B"]
        for i, (sp, sa) in enumerate(self.round_scores[:2]):
            if sp > sa:
                outcome = "Pemain Menang"
                c = (100, 255, 100)
            elif sa > sp:
                outcome = "AI Menang"
                c = (255, 100, 100)
            else:
                outcome = "Seri"
                c = (200, 200, 100)

            row = font_score.render(f"{labels[i]}  →  {sp} - {sa}  ({outcome})", True, c)
            screen.blit(row, row.get_rect(center=(width//2, y_score + i * 45)))

        # Divider
        pygame.draw.line(screen, (80, 80, 120), (panel_rect.left + 40, y_score + 105), (panel_rect.right - 40, y_score + 105), 2)

        note = font_note.render("Pilih AI mana yang terasa lebih menantang / lebih pintar:", True, (180, 180, 220))
        screen.blit(note, note.get_rect(center=(width//2, y_score + 130)))

        # Vote buttons
        mouse_pos = pygame.mouse.get_pos()
        self.vote_buttons = []

        btn_data = [
            ("Model A", (50, 50, 200), (100, 100, 255)),
            ("Model B", (180, 50, 50), (255, 100, 100)),
        ]

        btn_w, btn_h = 200, 65
        gap = 60
        total_w = btn_w * 2 + gap
        start_x = width//2 - total_w//2
        btn_y = y_score + 175

        for i, (label, color_base, color_hover) in enumerate(btn_data):
            bx = start_x + i * (btn_w + gap)
            rect = pygame.Rect(bx, btn_y, btn_w, btn_h)

            if rect.collidepoint(mouse_pos):
                col = color_hover
                pygame.draw.rect(screen, col, rect, border_radius=14)
                pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=14)
            else:
                col = color_base
                pygame.draw.rect(screen, col, rect, border_radius=14)

            txt = font_btn.render(label, True, (255, 255, 255))
            screen.blit(txt, txt.get_rect(center=rect.center))

            self.vote_buttons.append((rect, label))

        # hint
        hint = font_note.render("Pilihanmu akan membantu pengembangan AI!", True, (100, 180, 100))
        summary = self.stats.get("summary", {})
        total_sessions = self.stats.get("total_sessions", 0)
        if total_sessions > 0:
            before_pct = summary.get("before_rlhf_pct", 0.0)
            after_pct  = summary.get("after_rlhf_pct",  0.0)
            stat_font = pygame.font.SysFont("Calibri", 26)
            stat_text = f"Statistik kumulatif ({total_sessions} sesi) — Before RLHF: {before_pct}%  |  After RLHF: {after_pct}%"
            stat_render = stat_font.render(stat_text, True, (160, 220, 160))
            screen.blit(stat_render, stat_render.get_rect(center=(width//2, btn_y + btn_h + 70)))
        screen.blit(hint, hint.get_rect(center=(width//2, btn_y + btn_h + 35)))

    def draw_countdown(self, screen):
        width, height = screen.get_size()

        # overlay gelap
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        font = pygame.font.SysFont("Calibri", 120, bold=True)

        if self.countdown_value > 0:
            text = str(self.countdown_value)

            # render dulu
            render = font.render(text, True, (255, 255, 255))

            # efek pulse (membesar-mengecil)
            scale = 1 + 0.2 * math.sin(pygame.time.get_ticks() * 0.01)

            new_w = int(render.get_width() * scale)
            new_h = int(render.get_height() * scale)

            render = pygame.transform.smoothscale(render, (new_w, new_h))

        else:
            text = "GO!"
            render = font.render(text, True, (255, 255, 255))

        screen.blit(render, render.get_rect(center=(width//2, height//2)))

    def draw_pause_menu(self, screen):
        width, height = screen.get_size()
        mouse_pos = pygame.mouse.get_pos()
        self.pause_buttons = []

        # DARK OVERLAY (FULLSCREEN)
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        # PANEL (BIAR KELIHATAN MODERN)
        panel_width = int(width * 0.35)
        panel_height = int(height * 0.45)

        panel_rect = pygame.Rect(
            width // 2 - panel_width // 2,
            height // 2 - panel_height // 2,
            panel_width,
            panel_height
        )

        # background panel
        pygame.draw.rect(screen, (30, 30, 30), panel_rect, border_radius=20)

        # glow border
        pygame.draw.rect(screen, (0, 200, 255), panel_rect, 3, border_radius=20)

        # TITLE
        font_title = pygame.font.SysFont("Calibri", 70, bold=True)

        title = font_title.render("PAUSED", True, (255, 255, 255))
        title_rect = title.get_rect(center=(width // 2, panel_rect.top + 80))

        screen.blit(title, title_rect)

        # MENU OPTIONS
        font_menu = pygame.font.SysFont("Calibri", 45)

        for i, option in enumerate(self.menu_options):

            rect = pygame.Rect(
                width // 2 - 150,
                panel_rect.top + 150 + i * 70 - 25,
                300,
                50
            )

            # HOVER DETECTION
            if rect.collidepoint(mouse_pos):
                color = (255, 215, 0)  # kuning
            else:
                color = (180, 180, 180)

            text = font_menu.render(option, True, color)
            screen.blit(text, text.get_rect(center=rect.center))

            self.pause_buttons.append((rect, option))

    def draw_gradient_bg(self, screen, top_color, bottom_color):
        width, height = screen.get_size()
        for y in range(height):
            ratio = y / height
            r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
            g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
            b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (0, y), (width, y))


    def draw_bold_text(self, surface, text, font, x, y, color):
        offsets = [(0,0), (1,0), (0,1), (1,1)]

        for dx, dy in offsets:
            text_surface = font.render(text, True, color)
            surface.blit(text_surface, (x + dx, y + dy))

    def draw_neon_text(self, surface, text, font, x, y, main_color, glow_color):
        glow_surface = font.render(text, True, glow_color)
        glow_surface.set_alpha(70)

        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            surface.blit(glow_surface, (x+dx, y+dy))

        for dx, dy in [(0,0),(1,0),(0,1),(1,1)]:
            text_surface = font.render(text, True, main_color)
            surface.blit(text_surface, (x+dx, y+dy))

    def draw_glow_rounded_rect(self, surface, color, rect, width=7, glow_radius=1, border_radius=10):
        for i in range(glow_radius, 0, -1):
            alpha = int(255 * (i / glow_radius) ** 2)
            glow_color = (*color, alpha)

            glow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

            expanded_rect = rect.inflate(i*2, i*2)

            pygame.draw.rect(
                glow_surf,
                glow_color,
                expanded_rect,
                width + i,
                border_radius=border_radius + i
            )

            surface.blit(glow_surf, (0, 0))

        # core (main rectangle)
        pygame.draw.rect(
            surface,
            color,
            rect,
            width,
            border_radius=border_radius
        )

    def draw_glow_bar(self, surface, color, start, end, thickness=5):

        start = self.to_screen(start)
        end = self.to_screen(end)

        if start[1] == end[1]:  # horizontal
            x1, x2 = sorted([start[0], end[0]])

            rect = pygame.Rect(
                x1,
                start[1] - thickness // 2,
                x2 - x1,
                thickness
            )

        elif start[0] == end[0]:  # vertical
            y1, y2 = sorted([start[1], end[1]])

            rect = pygame.Rect(
                start[0] - thickness // 2,
                y1,
                thickness,
                y2 - y1
            )

        else:
            return

        self.draw_glow_rounded_rect(
            surface,
            color,
            rect,
            width=0,
            border_radius=int(thickness * self.scale / 2)
        )

    def draw_hexagon(self, surface, color, center, size, width=3):

        cx, cy = center

        # scaled_size = size * self.scale
        scaled_size = size

        points = []

        for i in range(6):
            angle = math.radians(60 * i)

            x = cx + scaled_size * math.cos(angle)
            y = cy + scaled_size * math.sin(angle)

            points.append((x, y))

        pygame.draw.polygon(surface, color, points, width)

    def draw_hex_grid(self, surface, size):
        hex_height = math.sqrt(3) * size
        hex_width = 2 * size

        vert_dist = hex_height
        horiz_dist = 1.5 * size

        table_w = self.env.WIDTH
        table_h = self.env.HEIGHT

        for row in range(int(table_h / vert_dist) + 2):
            for col in range(int(table_w / horiz_dist) + 2):

                x = col * horiz_dist
                y = row * vert_dist

                if col % 2 == 1:
                    y += vert_dist / 2

                # convert ke screen space
                screen_center = [x, y]

                # size ikut scaling
                scaled_size = size * self.scale

                self.draw_hexagon(
                    surface,
                    (15, 15, 15),
                    screen_center,
                    scaled_size,
                    3
                )

    def draw(self):
        self.screen.fill((0, 0, 0))
        # BACKGROUND
        self.screen.blit(self.hex_bg_scaled, (self.offset_x, self.offset_y))

        # CENTER LINE
        pygame.draw.line(
            self.screen,
            (128, 128, 128),
            self.to_screen([self.WIDTH / 2, self.TABLE_MARGIN]),
            self.to_screen([self.WIDTH / 2, self.HEIGHT - self.TABLE_MARGIN]),
            5
        )

        # UI TEXT
        minutes = self.time_left // 60
        seconds = self.time_left % 60

        timer_text = f"{minutes:02} : {seconds:02}"
        score_text = f"{self.score_player}    {self.score_ai}"

        self.draw_neon_text(self.screen, score_text, self.font, self.screen_width//2 - 30, 80, WHITE, BLUE)
        self.draw_neon_text(self.screen,timer_text, self.font, self.screen_width//2 - 41, 40, WHITE, RED)

        

        # CIRCLE CENTER TABLE
        pygame.draw.circle(
            self.screen,
            (128, 128, 128),
            self.to_screen([self.WIDTH/2, self.HEIGHT/2]),
            int(70 * self.scale),
            5
        )

        # GOAL
        pygame.draw.circle(
            self.screen,
            (128, 128, 128),
            self.to_screen([self.TABLE_MARGIN, self.HEIGHT/2]),
            int(self.GOAL_RADIUS * self.scale),
            5
        )

        pygame.draw.circle(
            self.screen,
            (128, 128, 128),
            self.to_screen([self.WIDTH - self.TABLE_MARGIN, self.HEIGHT/2]),
            int(self.GOAL_RADIUS * self.scale),
            5
        )

        # PLAYER
        self.screen.blit(
            self.player_img,
            self.player_img.get_rect(center=self.to_screen(self.env.player_pos))
        )

        # AI
        self.screen.blit(
            self.enemy_img,
            self.enemy_img.get_rect(center=self.to_screen(self.env.ai_pos))
        )

        # PUCK
        self.screen.blit(
            self.puck_img,
            self.puck_img.get_rect(center=self.to_screen(self.env.puck_pos))
        )

        # TOP
        self.draw_glow_bar(
            self.screen, (255, 80, 80),
            [self.WIDTH/2 + 10, self.TABLE_MARGIN],
            [self.WIDTH - self.TABLE_MARGIN, self.TABLE_MARGIN],
            5
        )
        self.draw_glow_bar(
            self.screen, (0, 255, 255),
            [self.TABLE_MARGIN, self.TABLE_MARGIN],
            [self.WIDTH/2 - 10, self.TABLE_MARGIN],
            5
        )

        # BOTTOM
        self.draw_glow_bar(
            self.screen, (255, 80, 80),
            [self.WIDTH/2 + 10, self.HEIGHT - self.TABLE_MARGIN],
            [self.WIDTH - self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN],
            5
        )

        self.draw_glow_bar(
            self.screen, (0, 255, 255),
            [self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN],
            [self.WIDTH/2 - 10, self.HEIGHT - self.TABLE_MARGIN],
            5
        )

        # LEFT
        self.draw_glow_bar(
            self.screen, (0, 255, 255),
            [self.TABLE_MARGIN, self.TABLE_MARGIN],
            [self.TABLE_MARGIN, self.HEIGHT/2 - self.GOAL_RADIUS],
            5
        )

        self.draw_glow_bar(
            self.screen, (0, 255, 255),
            [self.TABLE_MARGIN, self.HEIGHT/2 + self.GOAL_RADIUS],
            [self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN],
            5
        )

        # RIGHT
        self.draw_glow_bar(
            self.screen, (255, 80, 80),
            [self.WIDTH - self.TABLE_MARGIN, self.TABLE_MARGIN],
            [self.WIDTH - self.TABLE_MARGIN, self.HEIGHT/2 - self.GOAL_RADIUS],
            5
        )

        self.draw_glow_bar(
            self.screen, (255, 80, 80),
            [self.WIDTH - self.TABLE_MARGIN, self.HEIGHT/2 + self.GOAL_RADIUS],
            [self.WIDTH - self.TABLE_MARGIN, self.HEIGHT - self.TABLE_MARGIN],
            5
        )

        # PAUSE MENU
        if self.state == "paused":
            self.draw_pause_menu(self.screen)

        elif self.state == "menu":
            self.draw_main_menu(self.screen)

        elif self.state == "countdown":
            self.draw_countdown(self.screen)
        
        elif self.state == "game_over":
            self.draw_game_over(self.screen)

        elif self.state == "vote":
            self.draw_vote_screen(self.screen)

            

        pygame.display.flip()




if __name__ == "__main__":
    game = AirHockeyGame()
    game.run()