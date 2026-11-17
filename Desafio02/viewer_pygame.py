"""
viewer_pygame.py
Visualizador em janela (pygame) para o BigTaxiEnv: roda a política passo a passo,
com pausa, zoom, controle de velocidade e painel de métricas.

Exemplos:
  python viewer_pygame.py --model dqn_taxi_best.pt --size 80            # modelo treinado
  python viewer_pygame.py --model dqn_taxi_best.pt --no-guard           # sem proteção anti-travamento
O visualizador detecta sozinho se o modelo usa dueling e memória.
  python viewer_pygame.py --policy optimal --size 80                    # caminho ótimo (BFS), referência
  python viewer_pygame.py --policy random --size 80                     # agente aleatório

Teclas:
  ESPAÇO pausa/continua | seta direita: um passo (em pausa) | setas cima/baixo: velocidade | +/- zoom
  R novo episódio | T rastro | V janela de visão da rede | Q ou ESC sai
"""
import argparse
import os
import time

import numpy as np
import pygame

from big_taxi_env import BigTaxiEnv
from safe_policy import SafePolicy

ACTIONS = ["Sul", "Norte", "Leste", "Oeste", "Apanhar", "Largar"]
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
PANEL_W = 310

COL = dict(bg=(245, 245, 240), grid=(228, 228, 222), wall=(40, 40, 50),
           taxi=(245, 190, 40), taxi_edge=(120, 80, 10), pick=(214, 64, 64),
           dest=(60, 160, 80), trail=(150, 190, 235), view=(70, 110, 220),
           panel=(30, 34, 44), text=(235, 238, 245), sub=(160, 168, 184),
           ok=(110, 220, 140), bad=(240, 110, 110))


# ------------------------------------------------------------- políticas
def make_policy(args, env, sd):
    n_actions = env.action_space.n

    if args.policy == "random":
        return lambda obs: int(np.random.randint(n_actions))

    if args.policy == "optimal" and getattr(env, "is_multi", False):  # rota ótima p/ vários passageiros
        return lambda obs: env.optimal_action()

    if args.policy == "optimal":  # segue o gradiente do BFS (referência ótima)
        def act(obs):
            r, c = env.taxi
            if not env.carrying and env.taxi == env.pick:
                return 4
            if env.carrying and env.taxi == env.dest:
                return 5
            d = env.dist_dest if env.carrying else env.dist_pick
            for a, (dr, dc) in enumerate(MOVES):
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.size and 0 <= nc < env.size and 0 <= d[nr, nc] < d[r, c]:
                    return a
            return 0
        return act

    import torch  # só é necessário para a política treinada
    from train_dqn import QNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNet(env.observation_space.shape[0], n_actions, dueling=args.dueling).to(device)
    net.load_state_dict(sd)
    net.eval()

    def q_fn(obs):
        with torch.no_grad():
            return net(torch.from_numpy(obs).unsqueeze(0).to(device))[0].cpu().numpy()

    if args.no_guard:
        return lambda obs: int(np.argmax(q_fn(obs)))
    return SafePolicy(q_fn, env.size, env.view, args.loop_penalty,
                      kind="multi" if getattr(env, "is_multi", False) else "single",
                      capacity=getattr(env, "capacity", 3))


# ---------------------------------------------------------------- desenho
def build_static(env, cell):
    n = env.size
    surf = pygame.Surface((n * cell, n * cell))
    surf.fill(COL["bg"])
    if cell >= 6:
        for i in range(n + 1):
            pygame.draw.line(surf, COL["grid"], (i * cell, 0), (i * cell, n * cell))
            pygame.draw.line(surf, COL["grid"], (0, i * cell), (n * cell, i * cell))
    for r, c in np.argwhere(env.walls):
        pygame.draw.rect(surf, COL["wall"], (c * cell, r * cell, cell, cell))
    return surf


def cell_rect(pos, cell, inset=0):
    r, c = pos
    return pygame.Rect(c * cell + inset, r * cell + inset, cell - 2 * inset, cell - 2 * inset)


def draw_world(screen, static, env, cell, trail, show_trail, show_view, font):
    screen.blit(static, (0, 0))
    if show_trail:
        for pos in trail:
            pygame.draw.rect(screen, COL["trail"], cell_rect(pos, cell, max(1, cell // 3)))
    if not env.carrying:
        pygame.draw.rect(screen, COL["pick"], cell_rect(env.pick, cell))
        if cell >= 14:
            screen.blit(font.render("P", True, (255, 255, 255)),
                        (env.pick[1] * cell + cell // 3, env.pick[0] * cell + cell // 5))
    pygame.draw.rect(screen, COL["dest"], cell_rect(env.dest, cell))
    if cell >= 14:
        screen.blit(font.render("D", True, (255, 255, 255)),
                    (env.dest[1] * cell + cell // 3, env.dest[0] * cell + cell // 5))
    inset = max(1, cell // 8)
    pygame.draw.rect(screen, COL["taxi"], cell_rect(env.taxi, cell, inset), border_radius=max(1, cell // 4))
    pygame.draw.rect(screen, COL["taxi_edge"], cell_rect(env.taxi, cell, inset), width=1,
                     border_radius=max(1, cell // 4))
    if env.carrying:  # passageiro a bordo
        pygame.draw.rect(screen, COL["pick"], cell_rect(env.taxi, cell, max(2, cell // 3)))
    if cell < 14:  # com zoom pequeno, destaca os elementos com anéis
        rad = max(7, int(cell * 1.3))
        for pos, col, show in ((env.taxi, COL["taxi_edge"], True), (env.pick, COL["pick"], not env.carrying),
                               (env.dest, COL["dest"], True)):
            if show:
                center = (pos[1] * cell + cell // 2, pos[0] * cell + cell // 2)
                pygame.draw.circle(screen, col, center, rad, width=2)
    if show_view:  # área que a rede enxerga
        v = env.view
        r0, c0 = env.taxi[0] - v // 2, env.taxi[1] - v // 2
        pygame.draw.rect(screen, COL["view"], (c0 * cell, r0 * cell, v * cell, v * cell), width=2)


def draw_panel(screen, x0, h, font, small, lines, help_lines):
    pygame.draw.rect(screen, COL["panel"], (x0, 0, PANEL_W, h))
    y = 14
    for text, color, big in lines:
        f = font if big else small
        screen.blit(f.render(text, True, color), (x0 + 16, y))
        y += 30 if big else 22
    y = h - 14 - 20 * len(help_lines)
    for text in help_lines:
        screen.blit(small.render(text, True, COL["sub"]), (x0 + 16, y))
        y += 20


# ------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["model", "optimal", "random"], default="model")
    p.add_argument("--model", type=str, default="dqn_taxi_best.pt")
    p.add_argument("--dueling", action="store_true", help="(ignorado: detectado automaticamente)")
    p.add_argument("--no-guard", action="store_true", help="desliga a proteção anti-travamento")
    p.add_argument("--loop-penalty", type=float, default=0.3, help="penalidade de revisita da proteção")
    p.add_argument("--size", type=int, default=80)
    p.add_argument("--density", type=float, default=0.10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--view", type=int, default=11)
    p.add_argument("--seed", type=int, default=10_000, help="10000 = mesmos mapas da avaliação do treino")
    p.add_argument("--fps", type=int, default=30, help="passos por segundo")
    p.add_argument("--cell", type=int, default=0, help="pixels por célula (0 = automático)")
    p.add_argument("--max-frames", type=int, default=0, help="(teste) fecha após N quadros")
    p.add_argument("--screenshot", type=str, default=None, help="(teste) salva imagem ao fechar")
    args = p.parse_args()

    args.memory, sd = False, None
    if args.policy == "model":
        import torch
        from train_dqn import infer_config
        sd = torch.load(args.model, map_location="cpu")
        args.dueling, args.memory, kind = infer_config(sd, args.view)
        print(f"Modelo detectado: env={kind}, dueling={args.dueling}, memory={args.memory}")
        if kind == "multi":
            raise SystemExit("Este modelo é do ambiente multi-passageiro. Use: python viewer_cidade.py --model ...")

    env = BigTaxiEnv(size=args.size, obstacle_density=args.density, max_steps=args.max_steps,
                     view=args.view, shaping=False, memory=args.memory)
    policy = make_policy(args, env, sd)

    pygame.init()
    pygame.display.set_caption("Taxi DQN")
    font = pygame.font.Font(None, 28)
    small = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    desktop_h = pygame.display.Info().current_h or 900
    cell = args.cell or int(np.clip((desktop_h - 140) // args.size, 4, 14))

    def open_window(cell):
        w, h = args.size * cell + PANEL_W, max(args.size * cell, 460)
        return pygame.display.set_mode((w, h)), h

    screen, win_h = open_window(cell)

    # estado da sessão
    episode, n_success, ratios = 0, 0, []
    fps, paused, show_trail, show_view = args.fps, False, True, False
    step_once, running, frames = False, True, 0
    done_at = None

    def new_episode():
        nonlocal obs, info, trail, ep_return, steps, done, last_action, success, static, done_at, episode
        obs, info = env.reset(seed=args.seed + episode)
        episode += 1
        trail, ep_return, steps, done, last_action, success, done_at = [env.taxi], 0.0, 0, False, None, False, None
        static = build_static(env, cell)
        if hasattr(policy, "reset"):
            policy.reset()

    obs = info = static = None
    trail, ep_return, steps, done, last_action, success = [], 0.0, 0, False, None, False
    new_episode()

    while running:
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
                    fps = max(2, int(fps / 1.5))
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS):
                    grow = k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
                    cell = int(np.clip(cell + (2 if grow else -2), 4, 24))
                    screen, win_h = open_window(cell)
                    static = build_static(env, cell)
                elif k == pygame.K_r:
                    new_episode()
                elif k == pygame.K_t:
                    show_trail = not show_trail
                elif k == pygame.K_v:
                    show_view = not show_view

        # avança a simulação
        if not done and ((not paused) or step_once):
            last_action = policy(obs)
            obs, r, term, trunc, inf = env.step(last_action)
            ep_return += r
            steps += 1
            trail.append(env.taxi)
            if term or trunc:
                done, success = True, term
                done_at = time.time()
                if success:
                    n_success += 1
                    ratios.append(steps / info["optimal_steps"])
            step_once = False
        elif done and not paused and done_at and time.time() - done_at > 1.5:
            new_episode()  # próximo episódio automaticamente
        step_once = False

        # desenha
        draw_world(screen, static, env, cell, trail, show_trail, show_view, small)
        status = "a bordo" if env.carrying else "aguardando embarque"
        lines = [
            ("Taxi DQN", COL["text"], True),
            (f"Política: {args.policy}" + (" + guarda" if args.policy == "model" and not args.no_guard else ""),
             COL["sub"], False),
            ("", COL["text"], False),
            (f"Episódio: {episode}", COL["text"], False),
            (f"Passo: {steps} / {args.max_steps}", COL["text"], False),
            (f"Ótimo: {info['optimal_steps']} passos", COL["text"], False),
            (f"Retorno: {ep_return:.0f}", COL["text"], False),
            (f"Última ação: {ACTIONS[last_action] if last_action is not None else '-'}", COL["text"], False),
            (f"Passageiro: {status}", COL["text"], False),
            ("", COL["text"], False),
            (f"Sucessos: {n_success} / {episode - (0 if done else 1)}", COL["text"], False),
            (f"Passos / ótimo (média): {np.mean(ratios):.2f}" if ratios else "Passos / ótimo: -", COL["text"], False),
            ("", COL["text"], False),
            (f"Velocidade: {fps} passos/s" + ("  [PAUSA]" if paused else ""), COL["sub"], False),
            (f"Zoom: {cell} px/célula", COL["sub"], False),
        ]
        if done:
            msg = "ENTREGUE!" if success else "FALHOU (limite de passos)"
            lines.insert(3, (msg, COL["ok"] if success else COL["bad"], True))
        helps = ["ESPACO pausa | SETA DIR: 1 passo", "SETA CIMA/BAIXO velocidade",
                 "+/- zoom | R novo episodio", "T rastro | V visao | Q sai"]
        draw_panel(screen, args.size * cell, win_h, font, small, lines, helps)
        pygame.display.flip()

        clock.tick(fps if not done else 30)
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            running = False

    if args.screenshot:
        pygame.image.save(screen, args.screenshot)
    pygame.quit()


if __name__ == "__main__":
    main()
