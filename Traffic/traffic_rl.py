"""
MP 4 - Reinforcement Learning-Based Smart Traffic Signal Control System
=======================================================================
* Agent      : Q-Learning (tabular)  -> learns WHEN to switch the signal
* Environment: one 4-way junction with moving cars (written from scratch)
* Animation  : Pygame (cars move, lights glow, live statistics)

Run:
    pip install pygame
    python traffic_rl.py            # trains (first time only) + opens animation
    python traffic_rl.py --train    # force re-training
    python traffic_rl.py --no-gui   # only train + print AI vs Fixed-timer result

Keys in the animation:
    M  = switch AI / Fixed-timer      T = change traffic scenario
    UP / DOWN = traffic amount        LEFT / RIGHT = simulation speed
    SPACE = pause                     R = reset          ESC = quit
"""
import json
import math
import os
import random
import sys

# --------------------------------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------------------------------
W = H = 800            # map size in pixels
C = 400                # centre of the map
ROAD_HALF = 50         # half width of a road (one lane each direction)
STOP = 326             # position of the stop line (distance from spawn edge)
CAR_LEN, CAR_W = 26, 15
GAP = 8                # min gap between two cars
V_MAX, ACC, BRAKE = 75.0, 110.0, 160.0   # pixels/second based physics

MIN_GREEN = 6          # signal must stay green at least this long (sec)
YELLOW = 2.0
ALL_RED = 1.0
FIXED_GREEN = 15       # green time used by the old "fixed timer" system

LANES = ["North", "East", "South", "West"]   # lane index 0,1,2,3
Q_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")

SCENARIOS = {
    "Balanced traffic":    [0.25, 0.25, 0.25, 0.25],
    "Rush hour North-South": [0.45, 0.10, 0.45, 0.10],
    "Rush hour East-West":   [0.10, 0.45, 0.10, 0.45],
    "Light traffic":       [0.10, 0.10, 0.10, 0.10],
}

CAR_COLORS = [(231, 76, 60), (52, 152, 219), (241, 196, 15), (46, 204, 113),
              (155, 89, 182), (236, 240, 241), (230, 126, 34), (26, 188, 156)]


# --------------------------------------------------------------------------
# 2. ENVIRONMENT  (cars + traffic signal)
# --------------------------------------------------------------------------
class Car:
    __slots__ = ("s", "v", "color")

    def __init__(self, rng):
        self.s = 0.0                     # position of car FRONT along its lane
        self.v = V_MAX * 0.7             # speed
        self.color = rng.choice(CAR_COLORS)


class Intersection:
    """
    Lane 0 = cars coming from North, 1 = East, 2 = South, 3 = West.
    phase 0 -> North & South are green, phase 1 -> East & West are green.
    mode  'G' green, 'Y' yellow, 'AR' all-red (safety gap).
    """

    def __init__(self, rates, rng):
        self.rng = rng
        self.rates = list(rates)         # cars per second for each lane
        self.lanes = [[] for _ in range(4)]
        self.phase, self.mode = 0, "G"
        self.timer = 0.0
        self.green_t = 0.0
        self.t = 0.0
        self.passed = 0
        self.wait_sum = 0.0              # integral of (waiting cars) dt

    # ---------- signal ----------
    def light(self, i):
        group = i % 2                    # 0 -> N/S group, 1 -> E/W group
        if group != self.phase or self.mode == "AR":
            return "R"
        return "G" if self.mode == "G" else "Y"

    def request_switch(self):
        if self.mode == "G" and self.green_t >= MIN_GREEN:
            self.mode, self.timer = "Y", YELLOW
            return True
        return False

    # ---------- measurements ----------
    def waiting(self, i):
        return sum(1 for c in self.lanes[i] if c.s <= STOP + 2 and c.v < 8)

    def total_waiting(self):
        return sum(self.waiting(i) for i in range(4))

    def get_state(self):
        """State = (NS queue level, EW queue level, current phase, can-switch?)"""
        ns = self.waiting(0) + self.waiting(2)
        ew = self.waiting(1) + self.waiting(3)
        can = 1 if (self.mode == "G" and self.green_t >= MIN_GREEN) else 0
        return (min(ns // 3, 6), min(ew // 3, 6), self.phase, can)

    # ---------- physics ----------
    def step(self, dt):
        self.t += dt
        self.wait_sum += self.total_waiting() * dt

        # signal timers
        if self.mode == "G":
            self.green_t += dt
        else:
            self.timer -= dt
            if self.timer <= 0:
                if self.mode == "Y":
                    self.mode, self.timer = "AR", ALL_RED
                else:
                    self.phase, self.mode, self.green_t = 1 - self.phase, "G", 0.0

        # new cars arrive randomly
        for i in range(4):
            if self.rng.random() < self.rates[i] * dt:
                lane = self.lanes[i]
                if not lane or lane[-1].s - CAR_LEN > GAP + 6:
                    lane.append(Car(self.rng))

        # move cars
        for i, lane in enumerate(self.lanes):
            light = self.light(i)
            for k, car in enumerate(lane):
                limit = 1e9
                if k > 0:                                   # car in front
                    limit = lane[k - 1].s - CAR_LEN - GAP
                if car.s <= STOP and light != "G":          # red / yellow
                    dist = STOP - car.s
                    can_pass_yellow = light == "Y" and dist < 0.6 * car.v ** 2 / (2 * BRAKE)
                    if not can_pass_yellow:
                        limit = min(limit, STOP)
                dist = max(limit - car.s, 0.0)
                v_safe = math.sqrt(2 * BRAKE * dist)
                car.v = min(car.v + ACC * dt, V_MAX, v_safe)
                car.s = min(car.s + car.v * dt, max(limit, car.s))
            if lane and lane[0].s > W + CAR_LEN:
                lane.pop(0)
                self.passed += 1

    def advance(self, action, seconds=1.0, dt=0.2):
        """Apply action (0 keep, 1 switch) then simulate `seconds`. Returns reward."""
        switched = bool(action == 1 and self.request_switch())
        w0 = self.wait_sum
        for _ in range(int(round(seconds / dt))):
            self.step(dt)
        avg_wait = (self.wait_sum - w0) / seconds
        return -avg_wait / 5.0 - (0.3 if switched else 0.0)   # fewer waiting cars = better


# --------------------------------------------------------------------------
# 3. Q-LEARNING AGENT
# --------------------------------------------------------------------------
def best_action(Q, s):
    q = Q.get(s)
    return 0 if (q is None or q[0] >= q[1]) else 1


def random_rates(rng):
    ns, ew = rng.uniform(0.08, 0.5), rng.uniform(0.08, 0.5)
    return [ns * rng.uniform(0.8, 1.2), ew * rng.uniform(0.8, 1.2),
            ns * rng.uniform(0.8, 1.2), ew * rng.uniform(0.8, 1.2)]


def train(episodes=250, ep_seconds=240, alpha=0.15, gamma=0.9, seed=1):
    rng = random.Random(seed)
    Q = {}
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * episodes))        # exploration rate
        env = Intersection(random_rates(rng), rng)
        s = env.get_state()
        total = 0.0
        for sec in range(ep_seconds):
            if sec == ep_seconds // 2:                       # traffic pattern changes
                env.rates = random_rates(rng)
            q = Q.setdefault(s, [0.0, 0.0])
            a = rng.randint(0, 1) if rng.random() < eps else best_action(Q, s)
            r = env.advance(a)
            s2 = env.get_state()
            q2 = Q.setdefault(s2, [0.0, 0.0])
            q[a] += alpha * (r + gamma * max(q2) - q[a])     # Q-learning formula
            s, total = s2, total + r
        if (ep + 1) % 25 == 0:
            print(f"  episode {ep + 1:3d}/{episodes}   epsilon={eps:.2f}   reward={total:8.1f}")
    return Q


def save_q(Q):
    with open(Q_FILE, "w") as f:
        json.dump({",".join(map(str, k)): v for k, v in Q.items()}, f)


def load_q():
    with open(Q_FILE) as f:
        return {tuple(int(x) for x in k.split(",")): v for k, v in json.load(f).items()}


# --------------------------------------------------------------------------
# 4. CONTROLLERS + COMPARISON
# --------------------------------------------------------------------------
def controller(kind, env, Q):
    """Called once every simulated second."""
    if env.mode != "G":
        return
    if kind == "AI":
        a = best_action(Q, env.get_state())
    else:                                   # fixed timer
        a = 1 if env.green_t >= FIXED_GREEN else 0
    if a:
        env.request_switch()


def evaluate(Q, seconds=300):
    print("\nAI vs Fixed-timer (average waiting cars, lower is better)")
    print(f"{'Scenario':26s}{'Fixed':>8s}{'AI':>8s}")
    tot = {"Fixed": 0, "AI": 0}
    for name, rates in SCENARIOS.items():
        res = {}
        for kind in ("Fixed", "AI"):
            vals = []
            for seed in range(3):
                env = Intersection(rates, random.Random(100 + seed))
                for _ in range(seconds):
                    controller("AI" if kind == "AI" else "FIXED", env, Q)
                    env.advance(0)
                vals.append(env.wait_sum / env.t)
            res[kind] = sum(vals) / len(vals)
            tot[kind] += res[kind]
        print(f"{name:26s}{res['Fixed']:8.2f}{res['AI']:8.2f}")
    print(f"{'AVERAGE':26s}{tot['Fixed'] / 4:8.2f}{tot['AI'] / 4:8.2f}\n")


# --------------------------------------------------------------------------
# 5. PYGAME ANIMATION
# --------------------------------------------------------------------------
def run_visual(Q):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W + 280, H))
    pygame.display.set_caption("Smart Traffic Signal - Reinforcement Learning")
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("arial", 26, bold=True)
    f_mid = pygame.font.SysFont("arial", 18, bold=True)
    f_sml = pygame.font.SysFont("arial", 14)

    # ---------- static background ----------
    bg = pygame.Surface((W, H))
    bg.fill((24, 30, 38))
    deco = random.Random(7)
    for cx, cy in [(0, 0), (450, 0), (0, 450), (450, 450)]:       # city blocks
        pygame.draw.rect(bg, (44, 56, 68), (cx, cy, 350, 350))
        pygame.draw.rect(bg, (34, 82, 64), (cx + 14, cy + 14, 322, 322), border_radius=18)
        for _ in range(3):
            bx, by = cx + deco.randint(30, 220), cy + deco.randint(30, 220)
            bw, bh = deco.randint(60, 100), deco.randint(60, 100)
            pygame.draw.rect(bg, (18, 24, 30), (bx + 6, by + 6, bw, bh), border_radius=8)
            col = deco.choice([(86, 110, 140), (140, 100, 90), (110, 120, 150), (150, 140, 100)])
            pygame.draw.rect(bg, col, (bx, by, bw, bh), border_radius=8)
            pygame.draw.rect(bg, tuple(min(255, c + 30) for c in col), (bx + 8, by + 8, bw - 16, bh - 16), 2, border_radius=6)
        for _ in range(7):
            tx, ty = cx + deco.randint(25, 320), cy + deco.randint(25, 320)
            pygame.draw.circle(bg, (28, 120, 78), (tx, ty), deco.randint(7, 11))
    road = (58, 62, 72)
    pygame.draw.rect(bg, road, (C - ROAD_HALF, 0, 2 * ROAD_HALF, H))
    pygame.draw.rect(bg, road, (0, C - ROAD_HALF, W, 2 * ROAD_HALF))
    for d in range(0, 350, 24):                                    # centre dashes
        for a, b in [(0, 0), (450, 0)]:
            pygame.draw.rect(bg, (240, 200, 60), (C - 2, a + d, 4, 12))
            pygame.draw.rect(bg, (240, 200, 60), (a + d, C - 2, 12, 4))
    for d in range(0, 350, 24):
        pygame.draw.rect(bg, (240, 200, 60), (C - 2, 450 + d, 4, 12))
        pygame.draw.rect(bg, (240, 200, 60), (450 + d, C - 2, 12, 4))
    for k in range(C - ROAD_HALF + 5, C + ROAD_HALF - 5, 12):      # zebra crossings
        for y in (330, 452):
            pygame.draw.rect(bg, (225, 225, 225), (k, y, 7, 18))
        for x in (330, 452):
            pygame.draw.rect(bg, (225, 225, 225), (x, k, 18, 7))
    pygame.draw.rect(bg, (255, 255, 255), (C, STOP - 2, 50, 4))              # stop lines
    pygame.draw.rect(bg, (255, 255, 255), (C - 50, W - STOP - 2, 50, 4))
    pygame.draw.rect(bg, (255, 255, 255), (STOP - 2, C - 50, 4, 50))
    pygame.draw.rect(bg, (255, 255, 255), (W - STOP - 2, C, 4, 50))

    # ---------- car sprites ----------
    sprites = {}
    angle = {0: -90, 1: 180, 2: 90, 3: 0}

    def car_sprite(color, lane, braking):
        key = (color, lane, braking)
        if key not in sprites:
            s = pygame.Surface((CAR_LEN, CAR_W), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0, 0, CAR_LEN, CAR_W), border_radius=5)
            glass = (25, 30, 40)
            pygame.draw.rect(s, glass, (int(CAR_LEN * .58), 2, 6, CAR_W - 4), border_radius=2)
            pygame.draw.rect(s, glass, (int(CAR_LEN * .12), 2, 4, CAR_W - 4), border_radius=2)
            for y in (1, CAR_W - 4):
                pygame.draw.rect(s, (255, 240, 170), (CAR_LEN - 3, y, 3, 3))
                pygame.draw.rect(s, (255, 40, 40) if braking else (130, 20, 20), (0, y, 3, 3))
            sprites[key] = pygame.transform.rotate(s, angle[lane])
        return sprites[key]

    def car_xy(lane, s):
        sc = s - CAR_LEN / 2
        return [(C + 25, sc), (W - sc, C + 25), (C - 25, W - sc), (sc, C - 25)][lane]

    light_pos = [(C + 68, 296), (504, C + 68), (C - 68, 504), (296, C - 68)]
    lcol = {"R": (255, 60, 60), "Y": (255, 200, 40), "G": (60, 230, 110)}

    def draw_light(cx, cy, state):
        pygame.draw.rect(screen, (12, 12, 15), (cx - 10, cy - 28, 20, 56), border_radius=7)
        for n, key in enumerate("RYG"):
            pos = (cx, cy - 17 + n * 17)
            on = state == key
            pygame.draw.circle(screen, lcol[key] if on else tuple(c // 6 for c in lcol[key]), pos, 6)
            if on:
                glow = pygame.Surface((56, 56), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*lcol[key], 70), (28, 28), 24)
                screen.blit(glow, (pos[0] - 28, pos[1] - 28))

    # ---------- state ----------
    scen_names = list(SCENARIOS)
    scen, intensity, speed, kind, paused = 0, 1.0, 1, "AI", False
    history = []

    def new_env():
        history.clear()
        return Intersection([r * intensity for r in SCENARIOS[scen_names[scen]]], random.Random())

    env = new_env()
    decision_acc, hist_acc = 0.0, 0.0
    running = True

    while running:
        frame = min(clock.tick(60) / 1000.0, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_m:
                    kind = "FIXED" if kind == "AI" else "AI"
                    env = new_env()
                elif e.key == pygame.K_r:
                    env = new_env()
                elif e.key == pygame.K_t:
                    scen = (scen + 1) % len(scen_names)
                    env = new_env()
                elif e.key == pygame.K_UP:
                    intensity = min(2.0, intensity + 0.1)
                    env.rates = [r * intensity for r in SCENARIOS[scen_names[scen]]]
                elif e.key == pygame.K_DOWN:
                    intensity = max(0.3, intensity - 0.1)
                    env.rates = [r * intensity for r in SCENARIOS[scen_names[scen]]]
                elif e.key == pygame.K_RIGHT:
                    speed = min(8, speed * 2)
                elif e.key == pygame.K_LEFT:
                    speed = max(1, speed // 2)

        if not paused:
            sim = frame * speed
            n = max(1, int(sim / 0.05 + 0.999))
            h = sim / n
            for _ in range(n):
                env.step(h)
                decision_acc += h
                hist_acc += h
                if decision_acc >= 1.0:
                    decision_acc -= 1.0
                    controller(kind, env, Q)
                if hist_acc >= 1.0:
                    hist_acc -= 1.0
                    history.append(env.total_waiting())
                    del history[:-120]

        # ---------- draw map ----------
        screen.blit(bg, (0, 0))
        for i, lane in enumerate(env.lanes):
            for car in lane:
                x, y = car_xy(i, car.s)
                spr = car_sprite(car.color, i, car.v < 25 and car.s <= STOP + 2)
                screen.blit(spr, spr.get_rect(center=(x, y)))
        for i in range(4):
            draw_light(*light_pos[i], env.light(i))

        # ---------- side panel ----------
        px = W
        pygame.draw.rect(screen, (18, 22, 30), (px, 0, 280, H))
        pygame.draw.line(screen, (60, 70, 90), (px, 0), (px, H), 2)
        screen.blit(f_big.render("SMART TRAFFIC", True, (240, 240, 245)), (px + 18, 16))
        screen.blit(f_sml.render("Reinforcement Learning (Q-Learning)", True, (140, 150, 170)), (px + 18, 50))

        badge = (46, 160, 100) if kind == "AI" else (210, 120, 40)
        pygame.draw.rect(screen, badge, (px + 18, 80, 244, 34), border_radius=10)
        label = "CONTROL: AI AGENT" if kind == "AI" else "CONTROL: FIXED TIMER"
        screen.blit(f_mid.render(label, True, (255, 255, 255)), (px + 30, 87))

        screen.blit(f_sml.render("Scenario: " + scen_names[scen], True, (200, 205, 215)), (px + 18, 126))
        grp = "NORTH-SOUTH" if env.phase == 0 else "EAST-WEST"
        st = {"G": "GREEN", "Y": "YELLOW", "AR": "ALL RED"}[env.mode]
        col = {"G": (90, 235, 130), "Y": (255, 205, 60), "AR": (255, 90, 90)}[env.mode]
        screen.blit(f_mid.render(f"{grp}: {st}", True, col), (px + 18, 152))

        screen.blit(f_mid.render("Waiting cars", True, (240, 240, 245)), (px + 18, 192))
        for i, name in enumerate(LANES):
            wq = env.waiting(i)
            y = 222 + i * 30
            screen.blit(f_sml.render(name, True, (170, 178, 195)), (px + 18, y))
            pygame.draw.rect(screen, (40, 48, 62), (px + 80, y + 2, 180, 14), border_radius=7)
            bar = min(180, wq * 12)
            c = (90, 220, 130) if wq < 5 else (240, 190, 60) if wq < 9 else (240, 80, 80)
            if bar:
                pygame.draw.rect(screen, c, (px + 80, y + 2, bar, 14), border_radius=7)
            screen.blit(f_sml.render(str(wq), True, (255, 255, 255)), (px + 84, y + 2))

        avg = env.wait_sum / env.t if env.t > 1 else 0.0
        stats = [("Time", f"{int(env.t)} s"), ("Cars passed", str(env.passed)),
                 ("Avg waiting cars", f"{avg:.1f}"), ("Speed", f"x{speed}")]
        for i, (k, v) in enumerate(stats):
            y = 356 + i * 26
            screen.blit(f_sml.render(k, True, (150, 158, 178)), (px + 18, y))
            screen.blit(f_mid.render(v, True, (255, 255, 255)), (px + 170, y - 2))

        screen.blit(f_mid.render("Queue history", True, (240, 240, 245)), (px + 18, 476))
        gx, gy, gw, gh = px + 18, 506, 244, 110
        pygame.draw.rect(screen, (28, 34, 46), (gx, gy, gw, gh), border_radius=8)
        if len(history) > 1:
            top = max(10, max(history))
            pts = [(gx + 6 + i * (gw - 12) / 119, gy + gh - 8 - v / top * (gh - 16))
                   for i, v in enumerate(history)]
            pygame.draw.lines(screen, (90, 200, 255), False, pts, 2)

        tips = ["M  AI / Fixed timer", "T  traffic scenario", "UP/DOWN  traffic amount",
                "LEFT/RIGHT  speed", "SPACE pause   R reset"]
        for i, tline in enumerate(tips):
            screen.blit(f_sml.render(tline, True, (120, 130, 150)), (px + 18, 640 + i * 20))
        if paused:
            screen.blit(f_big.render("PAUSED", True, (255, 255, 255)), (W // 2 - 50, 20))

        pygame.display.flip()
    pygame.quit()


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if "--train" in sys.argv or not os.path.exists(Q_FILE):
        print("Training the Q-Learning agent (about a minute) ...")
        Q = train()
        save_q(Q)
        print(f"Saved Q-table with {len(Q)} states -> {Q_FILE}")
    else:
        Q = load_q()
        print("Loaded saved Q-table.")
    evaluate(Q)
    if "--no-gui" not in sys.argv:
        run_visual(Q)
