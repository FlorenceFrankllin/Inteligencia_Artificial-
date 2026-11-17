"""
viewer_cidade.py
Cenário urbano para o táxi: o agente é um táxi, os passageiros são pessoas acenando (cada uma com uma
cor e um número; o marcador de destino tem a mesma cor), as ruas são exatamente as células por onde o
táxi pode passar (asfalto com meio-fio) e os obstáculos são quarteirões: prédios, parques com árvores
e terrenos de obra com barreiras. Tudo é desenhado por código (sem imagens externas).

Exemplos:
  python viewer_cidade.py --model dqn_taxi_multi_best.pt --size 80   # modelo treinado (detecta tudo sozinho)
  python viewer_cidade.py --policy optimal --map streets             # só pelas ruas (malha de 1 célula)
  python viewer_cidade.py --model dqn_taxi_multi_streets_best.pt --map streets
  python viewer_cidade.py --policy optimal --size 80                 # rota ótima, 1 a 3 passageiros
  python viewer_cidade.py --policy optimal --max-passengers 1        # só 1 passageiro
  python viewer_cidade.py --model dqn_taxi_best.pt                   # modelo antigo de 1 passageiro

Teclas:
  ESPAÇO pausa | seta direita: um passo (em pausa) | setas cima/baixo: velocidade
  C alterna câmera (segue o táxi / mapa inteiro) | + e -: zoom (câmera que segue)
  R novo episódio | T rastro | V janela de visão da rede | Q ou ESC sai
"""
import argparse
import math
import time
from types import SimpleNamespace

import numpy as np
import pygame

from big_taxi_env import BigTaxiEnv
from multi_taxi_env import MultiTaxiEnv
from viewer_pygame import make_policy, draw_panel, ACTIONS, PANEL_W

BG = (18, 22, 30)
ROAD = [(62, 66, 76), (60, 64, 74)]
CURB = (214, 214, 206)
PALETTE = [(196, 120, 98), (110, 145, 190), (214, 190, 130), (150, 130, 170), (150, 172, 150)]
PCOLS = [(60, 130, 235), (236, 90, 165), (140, 100, 230)]        # cor de cada passageiro
PTEXT = [(130, 175, 255), (255, 145, 205), (190, 160, 255)]      # versão clara para o painel
HEADING = {0: 270, 1: 90, 2: 0, 3: 180}  # sul, norte, leste, oeste -> graus (carro desenhado olhando p/ leste)


def lighter(c, k):
    return tuple(min(255, v + k) for v in c)


def darker(c, k):
    return tuple(max(0, v - k) for v in c)


# ================================================================= sprites
class Sprites:
    """Todos os sprites de um tamanho de célula `s` (pré-renderizados)."""

    def __init__(self, s):
        self.s = s
        self.road = [self._road(i, False, False) for i in (0, 1)]
        self.road_h = self._road(0, True, False)
        self.road_v = self._road(0, False, True)
        self.buildings = [self._building(c) for c in PALETTE]
        self.tree = self._tree()
        self.barrier = self._barrier()
        self.persons = [self._person(c) for c in PCOLS]
        self.pins = [self._pin(c, i + 1) for i, c in enumerate(PCOLS)]
        self.pins_dim = []
        for p in self.pins:
            d = p.copy()
            d.set_alpha(105)
            self.pins_dim.append(d)
        self._cars, self._base_cars = {}, {}

    def car(self, angle, mask):
        """Táxi virado para `angle`, com as cabeças dos passageiros a bordo (índices em `mask`)."""
        key = (angle, mask)
        if key not in self._cars:
            if mask not in self._base_cars:
                self._base_cars[mask] = self._car(mask)
            self._cars[key] = pygame.transform.rotate(self._base_cars[mask], angle)
        return self._cars[key]

    def _road(self, shade, dash_h, dash_v):
        s = self.s
        surf = pygame.Surface((s, s))
        surf.fill(ROAD[shade])
        if s >= 14:
            t = max(1, int(s * 0.08))
            if dash_h:
                pygame.draw.rect(surf, (205, 205, 190), (int(s * 0.25), int(s * 0.5 - t / 2), int(s * 0.5), t))
            if dash_v:
                pygame.draw.rect(surf, (205, 205, 190), (int(s * 0.5 - t / 2), int(s * 0.25), t, int(s * 0.5)))
        return surf

    def _building(self, color):
        s = self.s
        surf = pygame.Surface((s, s))
        surf.fill((176, 176, 170))                                  # calçada
        pygame.draw.rect(surf, (128, 128, 124), (s * 0.14, s * 0.14, s * 0.8, s * 0.8))   # sombra
        pygame.draw.rect(surf, color, (s * 0.06, s * 0.06, s * 0.8, s * 0.8))
        pygame.draw.rect(surf, lighter(color, 38), (s * 0.16, s * 0.16, s * 0.6, s * 0.6))
        if s >= 14:
            pygame.draw.rect(surf, (190, 220, 240), (s * 0.28, s * 0.28, s * 0.2, s * 0.2))  # claraboia
            pygame.draw.rect(surf, (140, 140, 148), (s * 0.54, s * 0.52, s * 0.15, s * 0.15))  # ar-condicionado
        return surf

    def _tree(self):
        s = self.s
        surf = pygame.Surface((s, s))
        surf.fill((88, 146, 88))                                    # grama
        pygame.draw.circle(surf, (36, 96, 52), (s * 0.54, s * 0.56), s * 0.42)   # sombra da copa
        pygame.draw.circle(surf, (44, 112, 60), (s * 0.5, s * 0.5), s * 0.42)
        pygame.draw.circle(surf, (72, 152, 86), (s * 0.42, s * 0.42), s * 0.24)
        return surf

    def _barrier(self):
        s = self.s
        surf = pygame.Surface((s, s))
        surf.fill((172, 154, 124))                                   # terreno de obra (não é rua)
        for fx, fy in ((0.2, 0.2), (0.72, 0.22), (0.4, 0.78), (0.85, 0.7), (0.12, 0.62)):
            pygame.draw.circle(surf, (150, 132, 104), (s * fx, s * fy), max(1, s * 0.05))
        x0, y0, w, h = s * 0.08, s * 0.3, s * 0.84, s * 0.26
        pygame.draw.rect(surf, (30, 30, 34), (s * 0.18, y0 + h, s * 0.1, s * 0.14))     # pés
        pygame.draw.rect(surf, (30, 30, 34), (s * 0.72, y0 + h, s * 0.1, s * 0.14))
        for k in range(6):
            col = (240, 130, 30) if k % 2 == 0 else (245, 245, 245)
            pygame.draw.rect(surf, col, (x0 + k * w / 6, y0, w / 6 + 1, h))
        pygame.draw.rect(surf, (30, 30, 34), (x0, y0, w, h), 1)
        pygame.draw.polygon(surf, (240, 130, 30), [(s * 0.64, s * 0.92), (s * 0.84, s * 0.92), (s * 0.74, s * 0.66)])
        pygame.draw.rect(surf, (245, 245, 245), (s * 0.69, s * 0.78, s * 0.1, s * 0.05))
        return surf

    def _car(self, mask):
        s = self.s
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        L, W = s * 0.88, s * 0.56
        x0, y0 = (s - L) / 2, (s - W) / 2
        pygame.draw.ellipse(surf, (0, 0, 0, 80), (x0 + s * 0.03, y0 + s * 0.07, L, W))          # sombra
        wh, ww = s * 0.13, s * 0.22
        for fx in (0.12, 0.64):                                                                 # rodas
            for y in (y0 - wh * 0.35, y0 + W - wh * 0.65):
                pygame.draw.rect(surf, (24, 24, 28), (x0 + L * fx, y, ww, wh), border_radius=max(1, int(wh / 3)))
        body = (247, 196, 40)
        pygame.draw.rect(surf, body, (x0, y0, L, W), border_radius=max(2, int(s * 0.16)))
        pygame.draw.rect(surf, (170, 120, 10), (x0, y0, L, W), 1, border_radius=max(2, int(s * 0.16)))
        glass = (44, 66, 88)
        pygame.draw.polygon(surf, glass, [(x0 + L * 0.76, y0 + W * 0.2), (x0 + L * 0.58, y0 + W * 0.1),
                                          (x0 + L * 0.58, y0 + W * 0.9), (x0 + L * 0.76, y0 + W * 0.8)])
        pygame.draw.polygon(surf, glass, [(x0 + L * 0.16, y0 + W * 0.2), (x0 + L * 0.30, y0 + W * 0.1),
                                          (x0 + L * 0.30, y0 + W * 0.9), (x0 + L * 0.16, y0 + W * 0.8)])
        pygame.draw.rect(surf, (252, 216, 90), (x0 + L * 0.30, y0 + W * 0.1, L * 0.28, W * 0.8))  # teto
        pygame.draw.rect(surf, (250, 250, 250), (x0 + L * 0.38, y0 + W * 0.32, L * 0.13, W * 0.36))  # luminoso
        pygame.draw.rect(surf, (30, 30, 34), (x0 + L * 0.38, y0 + W * 0.32, L * 0.13, W * 0.36), 1)
        r = max(1, int(s * 0.045))
        for fy in (0.2, 0.8):
            pygame.draw.circle(surf, (255, 250, 200), (x0 + L * 0.97, y0 + W * fy), r)         # faróis
            pygame.draw.circle(surf, (220, 40, 40), (x0 + L * 0.03, y0 + W * fy), r)           # lanternas
        for i in mask:  # passageiros a bordo, vistos pela janela traseira (um boné na cor de cada um)
            cy = y0 + W * (0.28, 0.5, 0.72)[i]
            hr = max(2.0, W * 0.12)
            pygame.draw.circle(surf, PCOLS[i], (x0 + L * 0.23, cy), hr + max(1, s * 0.03))
            pygame.draw.circle(surf, (240, 200, 160), (x0 + L * 0.23, cy), hr)
        return surf

    def _person(self, shirt):
        s = self.s
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (0, 0, 0, 80), (s * 0.24, s * 0.5, s * 0.5, s * 0.3))        # sombra
        skin = (240, 200, 160)
        pygame.draw.line(surf, skin, (s * 0.7, s * 0.5), (s * 0.82, s * 0.2), max(1, int(s * 0.07)))  # braço acenando
        pygame.draw.circle(surf, skin, (s * 0.82, s * 0.18), max(1, s * 0.06))
        pygame.draw.ellipse(surf, shirt, (s * 0.26, s * 0.42, s * 0.46, s * 0.3))              # ombros
        pygame.draw.circle(surf, (60, 40, 30), (s * 0.49, s * 0.34), s * 0.16)                 # cabelo
        pygame.draw.circle(surf, skin, (s * 0.49, s * 0.38), s * 0.13)                         # cabeça
        return surf

    def _pin(self, color, label):
        s = self.s
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        d = darker(color, 70)
        pygame.draw.polygon(surf, color, [(s * 0.33, s * 0.5), (s * 0.67, s * 0.5), (s * 0.5, s * 0.86)])
        pygame.draw.circle(surf, color, (s * 0.5, s * 0.38), s * 0.24)
        pygame.draw.circle(surf, d, (s * 0.5, s * 0.38), s * 0.24, max(1, int(s * 0.04)))
        if s >= 18:
            pygame.draw.circle(surf, (255, 255, 255), (s * 0.5, s * 0.38), s * 0.15)
            txt = pygame.font.Font(None, max(12, int(s * 0.36))).render(str(label), True, d)
            surf.blit(txt, (s * 0.5 - txt.get_width() / 2, s * 0.38 - txt.get_height() / 2))
        else:
            pygame.draw.circle(surf, (255, 255, 255), (s * 0.5, s * 0.38), s * 0.1)
        return surf


def tile_kind(r, c):
    h = (r * 73856093 ^ c * 19349663) % 100
    return "building" if h < 55 else "tree" if h < 85 else "barrier"


def wall_kind(env, r, c):
    """(tipo, índice da paleta) de uma célula bloqueada."""
    if env.street_mask is None:                      # mapa aberto: obstáculos soltos
        return tile_kind(r, c), (r * 31 + c * 17) % len(PALETTE)
    if env.street_mask[r, c]:                        # obstáculo em cima de uma rua = obra
        return "barrier", 0
    S, (ro, co) = env.spacing_now, env.block_phase   # quarteirão inteiro com o mesmo estilo
    bi, bj = (r - ro) // S, (c - co) // S
    h = (bi * 73856093 ^ bj * 19349663) % 100
    return ("building" if h < 70 else "tree"), (bi * 7 + bj * 3) % len(PALETTE)


def build_static(env, sp, cell):
    """Ruas = exatamente as células por onde o táxi pode passar.
    O resto é quarteirão (prédio, parque, obra). Meio-fio onde a rua encosta no bloqueado."""
    n = env.size
    walls = env.walls
    ok = lambda r, c: 0 <= r < n and 0 <= c < n and not walls[r, c]
    surf = pygame.Surface((n * cell, n * cell))
    t = max(1, int(cell * 0.12))
    street_mode = env.street_mask is not None
    S, (ro, co) = env.spacing_now, env.block_phase
    for r in range(n):
        for c in range(n):
            x, y = c * cell, r * cell
            if walls[r, c]:
                k, pi = wall_kind(env, r, c)
                img = sp.buildings[pi] if k == "building" else sp.tree if k == "tree" else sp.barrier
                surf.blit(img, (x, y))
                continue
            if street_mode:   # faixa tracejada no meio de cada rua (só onde ela segue reta)
                dash_h = (r - ro) % S == 0 and (c - co) % S != 0 and ok(r, c - 1) and ok(r, c + 1)
                dash_v = (c - co) % S == 0 and (r - ro) % S != 0 and ok(r - 1, c) and ok(r + 1, c)
            else:
                dash_h = r % 4 == 2 and c % 4 != 2 and ok(r, c - 1) and ok(r, c + 1)
                dash_v = c % 4 == 2 and r % 4 != 2 and ok(r - 1, c) and ok(r + 1, c)
            img = sp.road_h if dash_h else sp.road_v if dash_v else sp.road[(r + c) % 2]
            surf.blit(img, (x, y))
            if not ok(r - 1, c):
                pygame.draw.rect(surf, CURB, (x, y, cell, t))
            if not ok(r + 1, c):
                pygame.draw.rect(surf, CURB, (x, y + cell - t, cell, t))
            if not ok(r, c - 1):
                pygame.draw.rect(surf, CURB, (x, y, t, cell))
            if not ok(r, c + 1):
                pygame.draw.rect(surf, CURB, (x + cell - t, y, t, cell))
    return surf.convert()


def build_minimap(env, ms):
    n = env.size
    surf = pygame.Surface((n * ms, n * ms))
    surf.fill((64, 68, 78))
    for r, c in np.argwhere(env.walls):
        pygame.draw.rect(surf, (30, 32, 38), (c * ms, r * ms, ms, ms))
    return surf


def cam_axis(center, view, total):
    """Deslocamento (px do mundo na borda esquerda/superior da janela)."""
    if total <= view:
        return -(view - total) / 2
    return float(np.clip(center - view / 2, 0, total - view))


def passengers(env):
    """Lista de (índice, origem, destino, status) com status 0 espera, 1 a bordo, 2 entregue."""
    if getattr(env, "is_multi", False):
        return [(i, env.picks[i], env.dests[i], env.status[i]) for i in range(env.n_p)]
    return [(0, env.pick, env.dest, 1 if env.carrying else 0)]


# ====================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["model", "optimal", "random"], default="model")
    p.add_argument("--model", type=str, default="dqn_taxi_multi_best.pt")
    p.add_argument("--env", choices=["auto", "multi", "single"], default="auto",
                   help="auto: usa o do modelo (ou multi para optimal/random)")
    p.add_argument("--capacity", type=int, default=3, help="passageiros simultâneos no táxi (multi)")
    p.add_argument("--min-passengers", type=int, default=1)
    p.add_argument("--max-passengers", type=int, default=3)
    p.add_argument("--map", choices=["open", "streets"], default="open",
                   help="open: obstáculos soltos | streets: só as ruas são transitáveis (repita o do treino)")
    p.add_argument("--street-spacing", type=int, default=4)
    p.add_argument("--random-spacing", action="store_true")
    p.add_argument("--street-density", type=float, default=0.04)
    p.add_argument("--dueling", action="store_true", help="(ignorado: detectado automaticamente)")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--loop-penalty", type=float, default=0.3)
    p.add_argument("--size", type=int, default=80)
    p.add_argument("--density", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--view", type=int, default=11)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--fps", type=int, default=12, help="passos por segundo")
    p.add_argument("--camera", choices=["follow", "map"], default="follow")
    p.add_argument("--cell-follow", type=int, default=36, help="pixels por célula na câmera que segue")
    p.add_argument("--vp", type=int, default=0, help="tamanho da área do mapa em pixels (0 = automático)")
    p.add_argument("--max-frames", type=int, default=0, help="(teste) fecha após N quadros")
    p.add_argument("--screenshot", type=str, default=None, help="(teste) salva imagem ao fechar")
    args = p.parse_args()

    args.memory, sd, kind = False, None, ("single" if args.env == "single" else "multi")
    if args.policy == "model":
        import torch
        from train_dqn import infer_config
        sd = torch.load(args.model, map_location="cpu")
        args.dueling, args.memory, kind = infer_config(sd, args.view)
        print(f"Modelo detectado: env={kind}, dueling={args.dueling}, memory={args.memory}")
    if args.max_steps is None:
        args.max_steps = 1200 if kind == "multi" else 600

    common = dict(size=args.size, obstacle_density=args.density, max_steps=args.max_steps,
                  view=args.view, shaping=False, memory=args.memory, map_mode=args.map,
                  street_spacing=args.street_spacing, random_spacing=args.random_spacing,
                  street_density=args.street_density)
    if kind == "multi":
        env = MultiTaxiEnv(capacity=args.capacity, min_passengers=args.min_passengers,
                           max_passengers=args.max_passengers, **common)
    else:
        env = BigTaxiEnv(**common)
    multi = kind == "multi"
    policy = make_policy(args, env, sd)
    n = env.size

    pygame.init()
    pygame.display.set_caption("Taxi DQN – Cidade")
    font, small, big = pygame.font.Font(None, 28), pygame.font.Font(None, 22), pygame.font.Font(None, 56)
    badge_font = pygame.font.Font(None, 20)
    desktop_h = pygame.display.Info().current_h or 1000
    VP = args.vp or int(np.clip(desktop_h - 120, 640, 900))
    screen = pygame.display.set_mode((VP + PANEL_W, VP))
    clock = pygame.time.Clock()

    cell_f = int(np.clip(args.cell_follow, 16, 48))
    cell_m = max(4, VP // n)
    mode = args.camera
    sprite_cache = {}
    ms = max(1, min(4, 250 // n))

    S = SimpleNamespace(episode=0, n_success=0, ratios=[], deliv_total=0, pass_total=0, obs=None, info=None,
                        trail=[], ret=0.0, steps=0, done=False, success=False, last_action=None, done_at=None,
                        heading=0, carpos=[0.0, 0.0], cam=[0.0, 0.0], static=None, mini=None, cell=cell_f)

    def cur_cell():
        return cell_f if mode == "follow" else cell_m

    def rebuild(snap=True):
        c = cur_cell()
        if c not in sprite_cache:
            sprite_cache[c] = Sprites(c)
        S.cell, S.sp = c, sprite_cache[c]
        S.static = build_static(env, S.sp, c)
        if snap:
            snap_camera()

    def car_center():
        return ((S.carpos[1] + 0.5) * S.cell, (S.carpos[0] + 0.5) * S.cell)

    def snap_camera():
        if mode == "follow":
            cx, cy = car_center()
        else:
            cx, cy = n * S.cell / 2, n * S.cell / 2
        S.cam = [cx, cy]

    def new_episode():
        S.obs, S.info = env.reset(seed=args.seed + S.episode)
        S.episode += 1
        S.trail, S.ret, S.steps, S.done, S.success = [env.taxi], 0.0, 0, False, False
        S.last_action, S.done_at, S.heading = None, None, 0
        S.carpos = [float(env.taxi[0]), float(env.taxi[1])]
        S.mini = build_minimap(env, ms)
        if hasattr(policy, "reset"):
            policy.reset()
        rebuild()

    def do_step():
        a = policy(S.obs)
        S.obs, r, term, trunc, inf = env.step(a)
        S.ret += r
        S.steps += 1
        S.trail.append(env.taxi)
        S.last_action = a
        if a < 4:
            S.heading = HEADING[a]
        if term or trunc:
            S.done, S.success, S.done_at = True, term, time.time()
            S.deliv_total += inf.get("delivered", int(term))
            S.pass_total += inf.get("n_passengers", 1)
            if term:
                S.n_success += 1
                S.ratios.append(S.steps / S.info["optimal_steps"])

    new_episode()
    fps, paused, show_trail, show_view = args.fps, False, True, False
    acc, step_once, running, frames = 0.0, False, True, 0

    while running:
        dt = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    step_once = True
                elif k == pygame.K_UP:
                    fps = min(480, int(fps * 1.5) + 1)
                elif k == pygame.K_DOWN:
                    fps = max(1, int(fps / 1.5))
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS):
                    grow = k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
                    cell_f = int(np.clip(cell_f + (4 if grow else -4), 16, 48))
                    mode = "follow"
                    rebuild()
                elif k == pygame.K_c:
                    mode = "map" if mode == "follow" else "follow"
                    rebuild()
                elif k == pygame.K_r:
                    new_episode()
                elif k == pygame.K_t:
                    show_trail = not show_trail
                elif k == pygame.K_v:
                    show_view = not show_view

        # ---- simulação (passos por segundo desacoplados dos 60 quadros/s)
        if not paused and not S.done:
            acc += dt * fps
            k = int(acc)
            acc -= k
            for _ in range(min(k, 40)):
                if S.done:
                    break
                do_step()
        elif paused and step_once and not S.done:
            do_step()
        step_once = False
        if S.done and not paused and S.done_at and time.time() - S.done_at > 1.8:
            new_episode()

        # ---- animação suave do carro e da câmera
        tgt = [float(env.taxi[0]), float(env.taxi[1])]
        if fps > 60:
            S.carpos = tgt
        else:
            f = 1 - math.exp(-dt * 18)
            S.carpos = [S.carpos[i] + (tgt[i] - S.carpos[i]) * f for i in (0, 1)]
        cell = S.cell
        total = n * cell
        if mode == "follow":
            cx, cy = car_center()
        else:
            cx, cy = total / 2, total / 2
        fc = 1 - math.exp(-dt * 9)
        S.cam = [S.cam[0] + (cx - S.cam[0]) * fc, S.cam[1] + (cy - S.cam[1]) * fc]
        ox, oy = cam_axis(S.cam[0], VP, total), cam_axis(S.cam[1], VP, total)

        # ---- desenho do mundo
        screen.fill(BG)
        screen.set_clip(pygame.Rect(0, 0, VP, VP))
        screen.blit(S.static, (round(-ox), round(-oy)))
        sp, t = S.sp, time.time()
        pax = passengers(env)

        def cpx(pos):  # canto superior esquerdo (tela) da célula (linha, coluna), aceita float
            return (pos[1] * cell - ox, pos[0] * cell - oy)

        def center(pos):
            x, y = cpx(pos)
            return (x + cell / 2, y + cell / 2)

        def badge(pos, i):  # numerozinho do passageiro
            if cell < 24:
                return
            x, y = cpx(pos)
            c = (x + cell * 0.14, y + cell * 0.16)
            pygame.draw.circle(screen, PCOLS[i], c, cell * 0.17)
            pygame.draw.circle(screen, (255, 255, 255), c, cell * 0.17, 1)
            tx = badge_font.render(str(i + 1), True, (255, 255, 255))
            screen.blit(tx, (c[0] - tx.get_width() / 2, c[1] - tx.get_height() / 2))

        if show_trail:
            rad = max(1, cell // 9)
            for pos in S.trail[-400:]:
                cx_, cy_ = center(pos)
                if -10 < cx_ < VP + 10 and -10 < cy_ < VP + 10:
                    pygame.draw.circle(screen, (225, 225, 232), (cx_, cy_), rad)

        pulse = 0.5 + 0.5 * math.sin(t * 4)
        ring_w = max(2, cell // 14)
        for i, pk, ds, st in pax:
            if st == 0:                                # pessoa esperando
                x, y = cpx(pk)
                screen.blit(sp.persons[i], (x, y))
                pygame.draw.circle(screen, PCOLS[i], center(pk), cell * (0.55 + 0.18 * pulse), ring_w)
                badge(pk, i)
            if st == 0:                                # destino ainda "apagado"
                x, y = cpx(ds)
                screen.blit(sp.pins_dim[i], (x, y))
            elif st == 1:                              # destino ativo
                x, y = cpx(ds)
                screen.blit(sp.pins[i], (x, y - cell * 0.08 * pulse))
                pygame.draw.circle(screen, PCOLS[i], center(ds), cell * (0.55 + 0.18 * (1 - pulse)), ring_w)

        mask = tuple(i for i, _, _, st in pax if st == 1)
        x, y = cpx(S.carpos)
        screen.blit(sp.car(S.heading, mask), (x, y))
        if cell < 16:                              # zoom pequeno: anéis de destaque
            screen_rings = [(S.carpos, (245, 190, 40))]
            for i, pk, ds, st in pax:
                if st == 0:
                    screen_rings.append((pk, PCOLS[i]))
                if st < 2:
                    screen_rings.append((ds, PCOLS[i]))
            for pos, col in screen_rings:
                pygame.draw.circle(screen, col, center(pos), max(7, int(cell * 1.3)), 2)

        if show_view:                              # área que a rede enxerga
            v = env.view
            r0, c0 = env.taxi[0] - v // 2, env.taxi[1] - v // 2
            pygame.draw.rect(screen, (70, 110, 220), (c0 * cell - ox, r0 * cell - oy, v * cell, v * cell), 2)

        # setas apontando para alvos fora da tela (câmera que segue)
        if mode == "follow":
            tx_, ty_ = center(S.carpos)
            for i, pk, ds, st in pax:
                if st == 2:
                    continue
                goal = pk if st == 0 else ds
                gx, gy = center(goal)
                if 0 <= gx <= VP and 0 <= gy <= VP:
                    continue
                dx, dy = gx - tx_, gy - ty_
                dist = math.hypot(dx, dy) or 1.0
                ux, uy = dx / dist, dy / dist
                px, py = float(np.clip(gx, 30, VP - 30)), float(np.clip(gy, 30, VP - 30))
                tri = [(px + ux * 16, py + uy * 16), (px - uy * 10 - ux * 8, py + ux * 10 - uy * 8),
                       (px + uy * 10 - ux * 8, py - ux * 10 - uy * 8)]
                pygame.draw.polygon(screen, PCOLS[i], tri)
                pygame.draw.polygon(screen, (255, 255, 255), tri, 2)
                d_cells = abs(goal[0] - env.taxi[0]) + abs(goal[1] - env.taxi[1])
                lab = small.render(f"{i + 1}: {d_cells}" if multi else str(d_cells), True, (255, 255, 255))
                screen.blit(lab, (px - lab.get_width() / 2 - ux * 34, py - lab.get_height() / 2 - uy * 34))

        # HUD sobre o mapa
        n_board = len(mask)
        if multi:
            n_deliv = sum(1 for *_, st in pax if st == 2)
            msg = f"Entregues {n_deliv}/{len(pax)}  |  a bordo {n_board}/{env.capacity}"
        else:
            msg = "Levando o passageiro ao destino" if n_board else "Indo buscar o passageiro"
        txt = small.render(msg, True, (255, 255, 255))
        hud = pygame.Surface((txt.get_width() + 20, 30), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 150))
        hud.blit(txt, (10, 6))
        screen.blit(hud, (12, 12))
        if S.done:
            if S.success:
                label = "TODOS ENTREGUES!" if multi else "ENTREGUE!"
            else:
                label = f"FALHOU: {env.delivered}/{env.n_p} entregues" if multi else "FALHOU (limite de passos)"
            banner = big.render(label, True, (110, 230, 150) if S.success else (240, 120, 120))
            bg = pygame.Surface((banner.get_width() + 40, banner.get_height() + 20), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 170))
            bg.blit(banner, (20, 10))
            screen.blit(bg, ((VP - bg.get_width()) // 2, 50))
        screen.set_clip(None)

        # ---- painel lateral
        W_ = (235, 238, 245)
        lines = [
            ("Taxi DQN – Cidade", W_, True),
            (f"Política: {args.policy}" + (" + guarda" if args.policy == "model" and not args.no_guard else ""),
             (160, 168, 184), False),
            ("", (0, 0, 0), False),
            (f"Episódio: {S.episode}", W_, False),
            (f"Passo: {S.steps} / {args.max_steps}", W_, False),
            (f"Ótimo: {S.info['optimal_steps']} passos", W_, False),
            (f"Retorno: {S.ret:.0f}", W_, False),
            (f"Última ação: {ACTIONS[S.last_action] if S.last_action is not None else '-'}", W_, False),
        ]
        names = {0: "aguardando", 1: "a bordo", 2: "entregue"}
        for i, pk, ds, st in pax:
            lines.append((f"Passageiro {i + 1}: {names[st]}", PTEXT[i], False))
        lines += [
            ("", (0, 0, 0), False),
            (f"Sucessos: {S.n_success} / {S.episode - (0 if S.done else 1)}", W_, False),
        ]
        if multi and S.pass_total:
            lines.append((f"Passageiros entregues: {S.deliv_total} / {S.pass_total}", W_, False))
        lines += [
            (f"Passos / ótimo (média): {np.mean(S.ratios):.2f}" if S.ratios else "Passos / ótimo: -", W_, False),
            ("", (0, 0, 0), False),
            (f"Velocidade: {fps} passos/s" + ("  [PAUSA]" if paused else ""), (160, 168, 184), False),
            (f"Câmera: {'segue o táxi' if mode == 'follow' else 'mapa inteiro'} ({cell}px)", (160, 168, 184), False),
        ]
        helps = ["ESPACO pausa | SETA DIR: 1 passo", "SETA CIMA/BAIXO velocidade",
                 "C camera | +/- zoom | R novo", "T rastro | V visao | Q sai"]
        draw_panel(screen, VP, VP, font, small, lines, helps)

        mm_size = n * ms
        mx, my = VP + (PANEL_W - mm_size) // 2, VP - 14 - 20 * len(helps) - 14 - mm_size
        if my > 440:                                # minimapa
            screen.blit(S.mini, (mx, my))
            if mode == "follow":
                pygame.draw.rect(screen, (255, 255, 255), (mx + ox / cell * ms, my + oy / cell * ms,
                                                           VP / cell * ms, VP / cell * ms), 1)
            for i, pk, ds, st in pax:
                if st == 0:
                    pygame.draw.rect(screen, PCOLS[i], (mx + pk[1] * ms - 1, my + pk[0] * ms - 1, 4, 4))
                if st < 2:
                    pygame.draw.rect(screen, PCOLS[i], (mx + ds[1] * ms - 1, my + ds[0] * ms - 1, 4, 4), 1)
            pygame.draw.rect(screen, (245, 190, 40), (mx + env.taxi[1] * ms - 1, my + env.taxi[0] * ms - 1, 4, 4))
            pygame.draw.rect(screen, (90, 96, 110), (mx, my, mm_size, mm_size), 1)

        pygame.display.flip()
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            running = False

    if args.screenshot:
        pygame.image.save(screen, args.screenshot)
    pygame.quit()


if __name__ == "__main__":
    main()
