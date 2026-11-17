"""
big_taxi_env.py
Taxi em grid grande (padrão 80x80) como ambiente Gymnasium customizado.

- Obstáculos aleatórios (células bloqueadas), sorteados a cada episódio (ou fixos).
- map_mode="streets": o táxi só anda pelas linhas das ruas (malha de 1 célula de largura, espaçamento
  `street_spacing`); todo o resto é quarteirão. Fase da malha, espaçamento (opcional) e obstáculos sobre as
  ruas são sorteados a cada mapa.
- Um passageiro com embarque e destino sorteados em células livres e alcançáveis (checado por BFS).
- Ações iguais ao Taxi-v3: 0 sul, 1 norte, 2 leste, 3 oeste, 4 apanhar, 5 largar.
- Recompensas iguais ao Taxi-v3: +20 entrega correta, -10 apanhar/largar ilegal, -1 por passo.
- Reward shaping opcional baseado em potencial (BFS), que preserva a política ótima.
- Curriculum: `curriculum_dist` limita a distância (caixa) entre táxi->embarque e embarque->destino.
- Observação compacta (vetor): janela local view x view (paredes + objetivo) e vetor até o objetivo.
- memory=True acrescenta uma janela de visitas recentes e a última ação (one-hot): permite à rede
  aprender a sair de loops, que uma política sem memória não consegue.
"""
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))  # sul, norte, leste, oeste (linha, coluna)


class BigTaxiEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, size=80, obstacle_density=0.10, max_steps=600, view=11,
                 shaping=True, gamma=0.99, new_map_every_episode=True,
                 curriculum_dist=None, memory=False, render_mode=None,
                 map_mode="open", street_spacing=4, random_spacing=False, street_density=0.04):
        super().__init__()
        assert view % 2 == 1, "view deve ser ímpar"
        self.size = size
        self.density = obstacle_density
        self.max_steps = max_steps
        self.view = view
        self.shaping = shaping
        self.gamma = gamma  # deve ser o mesmo gamma do agente
        self.new_map_every_episode = new_map_every_episode
        self.curriculum_dist = curriculum_dist if curriculum_dist is not None else 2 * size
        self.render_mode = render_mode
        self.memory = memory  # memória de curto prazo na observação (anti-loop)
        assert map_mode in ("open", "streets")
        self.map_mode = map_mode            # "open": obstáculos esparsos | "streets": malha de ruas
        self.street_spacing = street_spacing  # distância entre ruas paralelas (quarteirão = spacing-1)
        self.random_spacing = random_spacing  # sorteia o espaçamento (3 a 6) a cada mapa
        self.street_density = street_density  # obstáculos sorteados em cima das ruas
        self.street_mask, self.block_phase, self.spacing_now = None, (0, 0), street_spacing

        self.action_space = spaces.Discrete(6)
        obs_dim = 2 * view * view + 5
        if memory:
            obs_dim += view * view + 6  # janela de visitas + última ação (one-hot)
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)

        self.walls = None
        self._wall_pad = None

    # ------------------------------------------------------------------ mapa
    def _new_map(self):
        n = self.size
        rng = self.np_random
        if self.map_mode == "streets":
            # Só as linhas das ruas são transitáveis; o resto é quarteirão (obstáculo).
            # A posição da malha (fase) é sorteada a cada mapa; o espaçamento também, se pedido.
            S = int(rng.choice((3, 4, 5, 6))) if self.random_spacing else int(self.street_spacing)
            ro, co = int(rng.integers(S)), int(rng.integers(S))
            rows = ((np.arange(n) - ro) % S) == 0
            cols = ((np.arange(n) - co) % S) == 0
            street = rows[:, None] | cols[None, :]
            blocked_on_street = street & (rng.random((n, n)) < self.street_density)
            self.walls = (~street) | blocked_on_street
            self.street_mask, self.block_phase, self.spacing_now = street, (ro, co), S
        else:
            self.walls = rng.random((n, n)) < self.density
            self.street_mask = None
        pad = self.view // 2
        self._wall_pad = np.pad(self.walls, pad, constant_values=True).astype(np.float32)

    def _sample_free(self, center=None, radius=None):
        n = self.size
        rng = self.np_random
        for _ in range(200):
            if center is None or radius is None or radius >= 2 * n:
                r, c = int(rng.integers(n)), int(rng.integers(n))
            else:
                r = center[0] + int(rng.integers(-radius, radius + 1))
                c = center[1] + int(rng.integers(-radius, radius + 1))
                if not (0 <= r < n and 0 <= c < n):
                    continue
            if not self.walls[r, c]:
                return (r, c)
        free = np.argwhere(~self.walls)
        r, c = free[int(rng.integers(len(free)))]
        return (int(r), int(c))

    def _bfs(self, src):
        """Distância (em passos) de src a cada célula; -1 = inalcançável."""
        n = self.size
        walls = self.walls.tolist()
        dist = [[-1] * n for _ in range(n)]
        dist[src[0]][src[1]] = 0
        dq = deque([src])
        while dq:
            r, c = dq.popleft()
            d = dist[r][c] + 1
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not walls[nr][nc] and dist[nr][nc] < 0:
                    dist[nr][nc] = d
                    dq.append((nr, nc))
        return np.array(dist, dtype=np.int32)

    # ----------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.walls is None or self.new_map_every_episode:
            self._new_map()

        while True:
            for _ in range(100):
                taxi = self._sample_free()
                pick = self._sample_free(taxi, self.curriculum_dist)
                dest = self._sample_free(pick, self.curriculum_dist)
                if pick == dest:
                    continue
                dist_pick = self._bfs(pick)
                if dist_pick[taxi] < 0 or dist_pick[dest] < 0:
                    continue  # inalcançável: sorteia de novo
                break
            else:
                self._new_map()  # nenhuma configuração válida: gera outro mapa
                continue
            break

        self.taxi, self.pick, self.dest = taxi, pick, dest
        self.dist_pick = dist_pick
        self.dist_dest = self._bfs(dest)
        self.dist_pd = int(dist_pick[dest])
        self.carrying = False
        self.steps = 0
        pad = self.view // 2
        self._visit_pad = np.zeros((n_ := self.size + 2 * pad, n_), dtype=np.float32)
        self._visit_pad[taxi[0] + pad, taxi[1] + pad] = 1.0
        self.last_action = -1
        info = {"optimal_steps": int(dist_pick[taxi]) + 1 + self.dist_pd + 1}
        return self._obs(), info

    # ------------------------------------------------------------- potencial
    def _potential(self):
        """Estimativa (negativa) do que falta percorrer. Contínua ao embarcar."""
        if self.carrying:
            return -float(self.dist_dest[self.taxi])
        return -float(self.dist_pick[self.taxi] + self.dist_pd)

    # ------------------------------------------------------------------ step
    def step(self, action):
        self.steps += 1
        n = self.size
        phi_old = self._potential()
        reward, terminated = -1.0, False

        if action < 4:
            dr, dc = MOVES[action]
            nr, nc = self.taxi[0] + dr, self.taxi[1] + dc
            if 0 <= nr < n and 0 <= nc < n and not self.walls[nr, nc]:
                self.taxi = (nr, nc)
        elif action == 4:
            if (not self.carrying) and self.taxi == self.pick:
                self.carrying = True
            else:
                reward = -10.0
        else:
            if self.carrying and self.taxi == self.dest:
                reward, terminated = 20.0, True
            else:
                reward = -10.0

        pad = self.view // 2
        self._visit_pad[self.taxi[0] + pad, self.taxi[1] + pad] += 1.0
        self.last_action = int(action)

        raw = reward
        if self.shaping:
            phi_new = 0.0 if terminated else self._potential()
            reward += self.gamma * phi_new - phi_old

        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"raw_reward": raw, "success": terminated}
        return self._obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------- observação
    def _goal(self):
        return self.dest if self.carrying else self.pick

    def _obs(self):
        n, v = self.size, self.view
        r = v // 2
        tr, tc = self.taxi
        walls_win = self._wall_pad[tr:tr + v, tc:tc + v]
        gr, gc = self._goal()
        goal_win = np.zeros((v, v), dtype=np.float32)
        if abs(gr - tr) <= r and abs(gc - tc) <= r:
            goal_win[gr - tr + r, gc - tc + r] = 1.0
        vec = np.array([(gr - tr) / n, (gc - tc) / n, float(self.carrying),
                        tr / n, tc / n], dtype=np.float32)
        parts = [walls_win.ravel(), goal_win.ravel(), vec]
        if self.memory:
            visits = np.minimum(self._visit_pad[tr:tr + v, tc:tc + v], 4.0) / 4.0
            prev = np.zeros(6, dtype=np.float32)
            if self.last_action >= 0:
                prev[self.last_action] = 1.0
            parts += [visits.ravel(), prev]
        return np.concatenate(parts).astype(np.float32)

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "ansi":
            return self._ansi()
        if self.render_mode == "rgb_array":
            return self._rgb(cell=8)
        return None

    def _ansi(self):
        g = np.full((self.size, self.size), ".", dtype="<U1")
        g[self.walls] = "#"
        if not self.carrying:
            g[self.pick] = "P"
        g[self.dest] = "D"
        g[self.taxi] = "C" if self.carrying else "T"
        return "\n".join("".join(row) for row in g)

    def _rgb(self, cell=8):
        img = np.full((self.size, self.size, 3), 240, dtype=np.uint8)
        img[self.walls] = (40, 40, 50)
        if not self.carrying:
            img[self.pick] = (214, 64, 64)
        img[self.dest] = (60, 160, 80)
        img[self.taxi] = (245, 190, 40)
        return np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)


if __name__ == "__main__":
    env = BigTaxiEnv(size=20, max_steps=100, view=7)
    obs, info = env.reset(seed=0)
    env.render_mode = "ansi"
    print(env.render())
    print("obs dim:", obs.shape, "| passos ótimos:", info["optimal_steps"])
