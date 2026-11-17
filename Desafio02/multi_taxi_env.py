"""
multi_taxi_env.py
Taxi com 1 a 3 passageiros (sorteado a cada episódio) em grid grande com obstáculos.

Regras
- Cada passageiro nasce numa célula livre diferente e tem um destino diferente dos demais.
  Todas as origens e destinos são células distintas e alcançáveis (checado por BFS).
- O táxi pode levar vários passageiros ao mesmo tempo, até `capacity` (padrão 3).
- Ações iguais ao Taxi-v3: 0 sul, 1 norte, 2 leste, 3 oeste, 4 apanhar, 5 largar.
  - apanhar (4): embarca o passageiro que espera na célula atual (se houver vaga).
  - largar  (5): desembarca o passageiro a bordo cujo destino é a célula atual.
- Recompensas: +20 por passageiro entregue, -10 apanhar/largar ilegal, -1 por passo.
- O episódio termina quando todos são entregues (success) ou ao atingir max_steps (truncado).
- Reward shaping opcional (potencial = -(menor rota restante que atende a todos), calculada por
  BFS + busca exaustiva da ordem dos eventos). Preserva a política ótima e só é usado no treino.

Observação (vetor), na ordem:
  [janela paredes (v*v)] [janela origens de quem espera (v*v)] [janela destinos de quem está a bordo (v*v)]
  [linha/n, coluna/n, a_bordo/3]
  [3 slots x 6: espera, a_bordo, dx_origem, dy_origem, dx_destino, dy_destino]  (ordenados por proximidade)
  se memory=True: [janela de visitas (v*v)] [última ação one-hot (6)]
"""
import numpy as np
from gymnasium import spaces

from big_taxi_env import BigTaxiEnv, MOVES

MAXP = 3


class MultiTaxiEnv(BigTaxiEnv):
    is_multi = True

    def __init__(self, size=80, obstacle_density=0.10, max_steps=1200, view=11,
                 shaping=True, gamma=0.99, new_map_every_episode=True,
                 curriculum_dist=None, memory=False, capacity=3,
                 min_passengers=1, max_passengers=3, render_mode=None,
                 map_mode="open", street_spacing=4, random_spacing=False, street_density=0.04):
        super().__init__(size=size, obstacle_density=obstacle_density, max_steps=max_steps, view=view,
                         shaping=shaping, gamma=gamma, new_map_every_episode=new_map_every_episode,
                         curriculum_dist=curriculum_dist, memory=memory, render_mode=render_mode,
                         map_mode=map_mode, street_spacing=street_spacing, random_spacing=random_spacing,
                         street_density=street_density)
        assert 1 <= min_passengers <= max_passengers <= MAXP
        assert 1 <= capacity <= MAXP
        self.capacity = capacity
        self.min_p, self.max_p = min_passengers, max_passengers
        obs_dim = 3 * view * view + 3 + 6 * MAXP
        if memory:
            obs_dim += view * view + 6
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.n_p, self.status, self.picks, self.dests = 0, [], [], []

    # ----------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super(BigTaxiEnv, self).reset(seed=seed)  # gym.Env.reset (semente do np_random)
        if self.walls is None or self.new_map_every_episode:
            self._new_map()

        while True:
            self.n_p = int(self.np_random.integers(self.min_p, self.max_p + 1))
            found = False
            for _ in range(100):
                taxi = self._sample_free()
                used, picks, dests, ok = set(), [], [], True
                for _i in range(self.n_p):
                    c = self._sample_free(taxi, self.curriculum_dist)
                    if c in used:
                        ok = False
                        break
                    used.add(c)
                    picks.append(c)
                if not ok:
                    continue
                for i in range(self.n_p):
                    c = self._sample_free(picks[i], self.curriculum_dist)
                    if c in used:
                        ok = False
                        break
                    used.add(c)
                    dests.append(c)
                if not ok:
                    continue
                dist_taxi = self._bfs(taxi)
                if any(dist_taxi[c] < 0 for c in picks + dests):
                    continue
                found = True
                break
            if found:
                break
            self._new_map()

        self.taxi, self.picks, self.dests = taxi, picks, dests
        self.d_pick = [self._bfs(c) for c in picks]
        self.d_dest = [self._bfs(c) for c in dests]
        self.status = [0] * self.n_p          # 0 espera, 1 a bordo, 2 entregue
        self.delivered, self.steps = 0, 0
        pad = self.view // 2
        self._visit_pad = np.zeros((self.size + 2 * pad, self.size + 2 * pad), dtype=np.float32)
        self._visit_pad[taxi[0] + pad, taxi[1] + pad] = 1.0
        self.last_action = -1
        cost, _ = self._tour()
        self._phi = -float(cost) / 10.0
        info = {"optimal_steps": int(cost), "n_passengers": self.n_p}
        return self._obs(), info

    # ------------------------------------------------------------ rota ótima
    def _n_onboard(self):
        return sum(1 for s in self.status if s == 1)

    def _tour(self):
        """Menor nº de passos (movimentos + apanhar/largar) para atender a todos a partir do estado atual.
        Devolve (custo, primeiro_evento) com primeiro_evento = ('p'|'d', i)."""
        n, cap = self.n_p, self.capacity
        wait = sum(1 << i for i in range(n) if self.status[i] == 0)
        board = sum(1 << i for i in range(n) if self.status[i] == 1)
        nb0 = bin(board).count("1")
        dp, dd, picks, dests = self.d_pick, self.d_dest, self.picks, self.dests
        best = [10 ** 9, None]

        def rec(pos, wait, board, nb, cost, first):
            if cost >= best[0]:
                return
            if not wait and not board:
                best[0], best[1] = cost, first
                return
            for i in range(n):
                bit = 1 << i
                if board & bit:
                    rec(dests[i], wait, board ^ bit, nb - 1, cost + int(dd[i][pos]) + 1, first or ("d", i))
                elif (wait & bit) and nb < cap:
                    rec(picks[i], wait ^ bit, board | bit, nb + 1, cost + int(dp[i][pos]) + 1, first or ("p", i))

        rec(self.taxi, wait, board, nb0, 0, None)
        return (0, None) if best[1] is None else (best[0], best[1])

    def optimal_action(self):
        """Primeira ação de uma rota ótima (referência para o visualizador)."""
        _, first = self._tour()
        if first is None:
            return 0
        kind, i = first
        target, dmap, act = (self.dests[i], self.d_dest[i], 5) if kind == "d" else (self.picks[i], self.d_pick[i], 4)
        if self.taxi == target:
            return act
        r, c = self.taxi
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and 0 <= dmap[nr, nc] < dmap[r, c]:
                return a
        return 0

    # ------------------------------------------------------------------ step
    def step(self, action):
        self.steps += 1
        n = self.size
        reward = -1.0

        if action < 4:
            dr, dc = MOVES[action]
            nr, nc = self.taxi[0] + dr, self.taxi[1] + dc
            if 0 <= nr < n and 0 <= nc < n and not self.walls[nr, nc]:
                self.taxi = (nr, nc)
        elif action == 4:
            idx = next((i for i in range(self.n_p) if self.status[i] == 0 and self.picks[i] == self.taxi), None)
            if idx is not None and self._n_onboard() < self.capacity:
                self.status[idx] = 1
            else:
                reward = -10.0
        else:
            idx = next((i for i in range(self.n_p) if self.status[i] == 1 and self.dests[i] == self.taxi), None)
            if idx is not None:
                self.status[idx] = 2
                self.delivered += 1
                reward = 20.0
            else:
                reward = -10.0

        terminated = self.delivered == self.n_p
        pad = self.view // 2
        self._visit_pad[self.taxi[0] + pad, self.taxi[1] + pad] += 1.0
        self.last_action = int(action)

        raw = reward
        if self.shaping:
            phi_new = 0.0 if terminated else -float(self._tour()[0]) / 10.0
            reward += self.gamma * phi_new - self._phi
            self._phi = phi_new

        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"raw_reward": raw, "success": terminated, "delivered": self.delivered,
                "n_passengers": self.n_p}
        return self._obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------- observação
    def _obs(self):
        n, v = self.size, self.view
        r = v // 2
        tr, tc = self.taxi
        walls_win = self._wall_pad[tr:tr + v, tc:tc + v]
        pick_win = np.zeros((v, v), dtype=np.float32)
        dest_win = np.zeros((v, v), dtype=np.float32)

        slots = []
        for i in range(self.n_p):
            st = self.status[i]
            if st == 2:
                continue
            tgt = self.picks[i] if st == 0 else self.dests[i]
            if st == 0 and abs(self.picks[i][0] - tr) <= r and abs(self.picks[i][1] - tc) <= r:
                pick_win[self.picks[i][0] - tr + r, self.picks[i][1] - tc + r] = 1.0
            if st == 1 and abs(self.dests[i][0] - tr) <= r and abs(self.dests[i][1] - tc) <= r:
                dest_win[self.dests[i][0] - tr + r, self.dests[i][1] - tc + r] = 1.0
            key = (0 if st == 1 else 1, abs(tgt[0] - tr) + abs(tgt[1] - tc))
            vec = [float(st == 0), float(st == 1),
                   (self.picks[i][0] - tr) / n if st == 0 else 0.0,
                   (self.picks[i][1] - tc) / n if st == 0 else 0.0,
                   (self.dests[i][0] - tr) / n if st == 1 else 0.0,
                   (self.dests[i][1] - tc) / n if st == 1 else 0.0]
            slots.append((key, i, vec))
        slots.sort()
        slot_vec = np.zeros(6 * MAXP, dtype=np.float32)
        for k, (_, _, vec) in enumerate(slots):
            slot_vec[6 * k:6 * k + 6] = vec

        head = np.array([tr / n, tc / n, self._n_onboard() / MAXP], dtype=np.float32)
        parts = [walls_win.ravel(), pick_win.ravel(), dest_win.ravel(), head, slot_vec]
        if self.memory:
            visits = np.minimum(self._visit_pad[tr:tr + v, tc:tc + v], 4.0) / 4.0
            prev = np.zeros(6, dtype=np.float32)
            if self.last_action >= 0:
                prev[self.last_action] = 1.0
            parts += [visits.ravel(), prev]
        return np.concatenate(parts).astype(np.float32)

    # ---------------------------------------------------------------- render
    def _ansi(self):
        g = np.full((self.size, self.size), ".", dtype="<U1")
        g[self.walls] = "#"
        for i in range(self.n_p):
            if self.status[i] == 0:
                g[self.picks[i]] = "P"
            if self.status[i] < 2:
                g[self.dests[i]] = "D"
        g[self.taxi] = "T"
        return "\n".join("".join(row) for row in g)

    def _rgb(self, cell=8):
        img = np.full((self.size, self.size, 3), 240, dtype=np.uint8)
        img[self.walls] = (40, 40, 50)
        for i in range(self.n_p):
            if self.status[i] == 0:
                img[self.picks[i]] = (214, 64, 64)
            if self.status[i] < 2:
                img[self.dests[i]] = (60, 160, 80)
        img[self.taxi] = (245, 190, 40)
        return np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)
