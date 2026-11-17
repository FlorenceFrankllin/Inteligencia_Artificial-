"""
safe_policy.py
Proteções contra travamento para uma política gulosa sem memória.

Por que o táxi trava: a política é determinística. Se a ação preferida bate num obstáculo,
o estado não muda e a mesma ação é escolhida de novo, para sempre. O mesmo ocorre em loops
A -> B -> A. Todas as regras abaixo usam apenas informação que já está na observação do
agente (janela local de paredes, passageiros na janela, quantos estão a bordo, posição).

  1. Máscara de legalidade: não escolhe mover contra parede; só apanha/larga quando é possível.
  2. Ação forçada: se pode largar um passageiro (ou apanhar um), faz isso (é sempre vantajoso).
  3. Penalidade de revisita: subtrai `penalty` x (visitas à célula de destino neste episódio)
     do valor Q de cada movimento. Quebra loops, pois repetir um ciclo fica cada vez mais caro.

Dois formatos de observação são suportados:
  kind="single": BigTaxiEnv        (um passageiro)
  kind="multi":  MultiTaxiEnv      (1 a 3 passageiros)
"""
import numpy as np

MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))  # sul, norte, leste, oeste
MAXP = 3


# --------------------------------------------------------------- um passageiro
def _parts_single(obs, size, view):
    v = view
    base = 2 * v * v
    walls = obs[:v * v].reshape(v, v)
    goal = obs[v * v:base].reshape(v, v)
    carrying = obs[base + 2] > 0.5
    pos = (int(round(float(obs[base + 3]) * size)), int(round(float(obs[base + 4]) * size)))
    return walls, goal, carrying, pos


def legal_actions(obs, size, view):
    """Ações que fazem sentido no estado `obs` (ambiente de um passageiro)."""
    r = view // 2
    walls, goal, carrying, _ = _parts_single(obs, size, view)
    legal = np.zeros(6, dtype=bool)
    for a, (dr, dc) in enumerate(MOVES):
        legal[a] = walls[r + dr, r + dc] < 0.5
    at_goal = goal[r, r] > 0.5
    legal[4] = at_goal and not carrying
    legal[5] = at_goal and carrying
    if not legal.any():  # estado sem saída (não deveria ocorrer): libera os movimentos
        legal[:4] = True
    return legal


def _pos_single(obs, size, view):
    return _parts_single(obs, size, view)[3]


# ------------------------------------------------------------ vários passageiros
def legal_actions_multi(obs, size, view, capacity=3):
    v = view
    r = v // 2
    walls = obs[:v * v].reshape(v, v)
    picks = obs[v * v:2 * v * v].reshape(v, v)
    dests = obs[2 * v * v:3 * v * v].reshape(v, v)
    n_board = int(round(float(obs[3 * v * v + 2]) * MAXP))
    legal = np.zeros(6, dtype=bool)
    for a, (dr, dc) in enumerate(MOVES):
        legal[a] = walls[r + dr, r + dc] < 0.5
    legal[4] = bool(picks[r, r] > 0.5) and n_board < capacity
    legal[5] = bool(dests[r, r] > 0.5)
    if not legal.any():
        legal[:4] = True
    return legal


def _pos_multi(obs, size, view):
    b = 3 * view * view
    return (int(round(float(obs[b]) * size)), int(round(float(obs[b + 1]) * size)))


def legal_for(kind, obs, size, view, capacity=3):
    if kind == "multi":
        return legal_actions_multi(obs, size, view, capacity)
    return legal_actions(obs, size, view)


class SafePolicy:
    """Envolve uma função q_fn(obs) -> vetor de Q e devolve a ação segura."""

    def __init__(self, q_fn, size, view, penalty=0.3, kind="single", capacity=3):
        self.q_fn, self.size, self.view, self.penalty = q_fn, size, view, penalty
        self.kind, self.capacity = kind, capacity
        self.visits = {}

    def reset(self):
        self.visits = {}

    def __call__(self, obs):
        legal = legal_for(self.kind, obs, self.size, self.view, self.capacity)
        if legal[5]:   # pode largar: largar
            return 5
        if legal[4]:   # pode apanhar: apanhar
            return 4

        pos = (_pos_multi if self.kind == "multi" else _pos_single)(obs, self.size, self.view)
        q = np.array(self.q_fn(obs), dtype=np.float64)
        self.visits[pos] = self.visits.get(pos, 0) + 1
        for a, (dr, dc) in enumerate(MOVES):
            if legal[a]:
                q[a] -= self.penalty * self.visits.get((pos[0] + dr, pos[1] + dc), 0)
        q[~legal] = -np.inf
        return int(np.argmax(q))
