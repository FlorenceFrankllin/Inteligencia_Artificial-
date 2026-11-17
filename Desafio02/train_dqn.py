"""
train_dqn.py
DQN Essencial (Vanilla DQN + Reward Shaping + Currículo de Distância + Máscara de Ações).

Equilíbrio ideal para apresentação acadêmica:
- Mantém o algoritmo DQN clássico (sem Double DQN, sem Dueling, sem rede complexa e sem memória extra).
- Usa apenas os 3 ingredientes essenciais para alta taxa de sucesso em mapas novos:
  1. Reward Shaping via BFS (dá sinal contínuo de aproximação ao objetivo).
  2. Currículo de Distância (começa com entregas curtas e aumenta o raio gradualmente).
  3. Máscara de ações legais (evita colisões contra paredes na escolha da ação).

Exemplos:
  python train_dqn.py --size 20 --episodes 2000
  python train_dqn.py --size 80 --episodes 5000
  python train_dqn.py --size 80 --eval-only dqn_taxi_multi_essencial_best.pt --loop-penalty 0.3
"""
import argparse
import atexit
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from big_taxi_env import BigTaxiEnv
from multi_taxi_env import MultiTaxiEnv
from safe_policy import SafePolicy, legal_for
from monitor import TrainingMonitor, append_result


# ------------------------------------------------------------------ rede (MLP Simples)
class QNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=512, dueling=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.q = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.q(self.body(x))


# ---------------------------------------------------------------- replay
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.m2 = np.ones((capacity, 6), dtype=bool)

    def add(self, s, a, r, s2, terminated, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, terminated
        self.m2[i] = m2
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.n, size=batch)
        t = lambda x: torch.from_numpy(x[idx]).to(device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d), t(self.m2)


def infer_config(sd, view):
    """Descobre (dueling, memory, env) a partir dos pesos salvos."""
    in_dim = sd["body.0.weight"].shape[1]
    dueling = "v.weight" in sd
    v2 = view * view
    table = {2 * v2 + 5: (False, "single"), 3 * v2 + 11: (True, "single"),
             3 * v2 + 21: (False, "multi"), 4 * v2 + 27: (True, "multi")}
    if in_dim in table:
        memory, kind = table[in_dim]
        return dueling, memory, kind
    raise SystemExit(f"O modelo espera entrada de tamanho {in_dim}, que não bate com --view {view}. "
                     f"Esperado: {sorted(table)}. Use o mesmo --view do treino.")


def make_env(args, shaping, render_mode=None, training=False):
    common = dict(size=args.size, obstacle_density=args.density, max_steps=args.max_steps, view=args.view,
                  shaping=shaping, gamma=args.gamma, memory=False, render_mode=render_mode,
                  map_mode=args.map, street_spacing=args.street_spacing, random_spacing=args.random_spacing,
                  street_density=args.street_density,
                  new_map_every_episode=not (args.fixed_map and training))
    if args.env == "multi":
        return MultiTaxiEnv(capacity=args.capacity, min_passengers=args.min_passengers,
                            max_passengers=args.max_passengers, **common)
    return BigTaxiEnv(**common)


# ------------------------------------------------------------- avaliação
def evaluate(policy, args, device, n_episodes, seed, record=False):
    """Política gulosa em mapas novos, sem shaping (recompensa original)."""
    env = make_env(args, shaping=False, render_mode="rgb_array" if record else None)

    def q_fn(obs):
        with torch.no_grad():
            return policy(torch.from_numpy(obs).unsqueeze(0).to(device))[0].cpu().numpy()

    use_guard = (not args.no_mask) or args.loop_penalty > 0
    guard = SafePolicy(q_fn, args.size, args.view, args.loop_penalty, kind=args.env,
                       capacity=args.capacity) if use_guard else None
    returns, successes, ratios, frames, delivered, npass = [], [], [], [], 0, 0
    for k in range(n_episodes):
        obs, info = env.reset(seed=seed + k)
        optimal, ep_ret, done, steps = info["optimal_steps"], 0.0, False, 0
        if guard:
            guard.reset()
        while not done:
            action = guard(obs) if guard else int(np.argmax(q_fn(obs)))
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            steps += 1
            done = term or trunc
            if record and k == 0 and steps % 2 == 0:
                frames.append(env.render())
        returns.append(ep_ret)
        successes.append(float(term))
        delivered += info.get("delivered", int(term))
        npass += info.get("n_passengers", 1)
        if term:
            ratios.append(steps / optimal)
    stats = {
        "return": float(np.mean(returns)),
        "success": float(np.mean(successes)),
        "delivered_frac": delivered / max(1, npass),
        "steps_vs_optimal": float(np.mean(ratios)) if ratios else float("nan"),
    }
    return stats, frames


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=80)
    p.add_argument("--density", type=float, default=0.10)
    p.add_argument("--env", choices=["multi", "single"], default="multi")
    p.add_argument("--capacity", type=int, default=3)
    p.add_argument("--min-passengers", type=int, default=1)
    p.add_argument("--max-passengers", type=int, default=3)
    p.add_argument("--map", choices=["open", "streets"], default="open")
    p.add_argument("--street-spacing", type=int, default=4)
    p.add_argument("--random-spacing", action="store_true")
    p.add_argument("--street-density", type=float, default=0.04)
    p.add_argument("--fixed-map", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--view", type=int, default=11)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--start-learning", type=int, default=2_000)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-every", type=int, default=2_000)
    p.add_argument("--eps-steps", type=int, default=None,
                   help="passos para decair epsilon (padrão: proporcional aos episódios)")
    p.add_argument("--min-eps", type=float, default=0.05)
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--curriculum-frac", type=float, default=0.4)
    p.add_argument("--no-mask", action="store_true", help="desliga a máscara de ações legais")
    p.add_argument("--loop-penalty", type=float, default=0.3,
                   help="penalidade de revisita na avaliação (padrão: 0.3)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--final-episodes", type=int, default=50)
    p.add_argument("--save-results", type=str, default=None)
    p.add_argument("--eval-only", type=str, default=None)
    p.add_argument("--gif", type=str, default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_sd = None
    if args.eval_only:
        eval_sd = torch.load(args.eval_only, map_location=device)
        _, _, args.env = infer_config(eval_sd, args.view)
        print(f"Modelo DQN Essencial detectado: env={args.env}")
    if args.max_steps is None:
        args.max_steps = (1200 if args.size > 30 else 400) if args.env == "multi" else 600
    if args.eps_steps is None:
        args.eps_steps = int(args.episodes * args.max_steps * 0.45)

    tag = ("_multi" if args.env == "multi" else "") + ("_streets" if args.map == "streets" else "") + "_essencial"

    env = make_env(args, shaping=not args.no_shaping, training=True)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    policy = QNet(obs_dim, n_actions, hidden=512).to(device)

    if args.eval_only:
        policy.load_state_dict(eval_sd)
        stats, frames = evaluate(policy, args, device, args.final_episodes, seed=args.eval_seed,
                                 record=bool(args.gif))
        print("Avaliação (DQN Essencial):", stats)
        if args.save_results:
            lo, hi = args.min_passengers, args.max_passengers
            append_result(args.save_results, dict(
                model="final" if "final" in os.path.basename(args.eval_only) else "best",
                variant="dqn essencial",
                passengers=str(lo) if lo == hi else f"{lo}-{hi}", loop_penalty=args.loop_penalty,
                map=args.map, episodes=args.final_episodes, seed=args.eval_seed,
                success=stats["success"], delivered_frac=stats["delivered_frac"],
                steps_vs_optimal=stats["steps_vs_optimal"], **{"return": stats["return"]}))
            print("resultado acrescentado em", args.save_results)
        save_gif(frames, args.gif)
        return

    target = QNet(obs_dim, n_actions, hidden=512).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = optim.Adam(policy.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer, obs_dim)

    def get_mask(o):
        if args.no_mask:
            return np.ones(6, dtype=bool)
        return legal_for(args.env, o, args.size, args.view, args.capacity)

    def update():
        s, a, r, s2, d, m2 = buf.sample(args.batch, device)
        q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: a rede online escolhe a melhor ação legal e a rede-alvo avalia
            a2 = policy(s2).masked_fill(~m2, -1e9).argmax(dim=1, keepdim=True)
            next_q = target(s2).gather(1, a2).squeeze(1)
            y = r + args.gamma * (1.0 - d) * next_q
        loss = F.smooth_l1_loss(q, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        return float(loss.item()), float(q.mean().item())

    mon = TrainingMonitor(tag=tag)
    atexit.register(mon.finalize)
    log, total_steps, best_score, t0 = [], 0, (-1.0, -1.0), time.time()

    for ep in range(args.episodes):
        # Currículo de Distância: começa com entregas curtas (raio 5) e expande até o mapa todo
        if args.no_curriculum:
            env.curriculum_dist = 2 * args.size
        else:
            frac = min(1.0, ep / max(1, args.curriculum_frac * args.episodes))
            env.curriculum_dist = int(5 + frac * (2 * args.size - 5))

        obs, info = env.reset(seed=args.seed + ep)
        raw_return, done, steps, success = 0.0, False, 0, False
        legal = get_mask(obs)
        while not done:
            eps = max(args.min_eps, 1.0 - total_steps * (1.0 - args.min_eps) / args.eps_steps)
            if random.random() < eps:
                action = int(np.random.choice(np.flatnonzero(legal)))
            else:
                with torch.no_grad():
                    q = policy(torch.from_numpy(obs).unsqueeze(0).to(device))[0].cpu().numpy()
                q[~legal] = -np.inf
                action = int(np.argmax(q))

            obs2, r, term, trunc, info = env.step(action)
            legal2 = get_mask(obs2)
            buf.add(obs, action, r, obs2, float(term), legal2)
            obs, legal = obs2, legal2
            raw_return += info["raw_reward"]
            steps += 1
            total_steps += 1
            done = term or trunc
            success = term

            if buf.n >= args.start_learning and total_steps % args.train_every == 0:
                mon.record_update(*update())
            if total_steps % args.target_every == 0:
                target.load_state_dict(policy.state_dict())

        log.append((ep + 1, raw_return, float(success), steps, eps, env.curriculum_dist,
                    info.get("delivered", int(success)), info.get("n_passengers", 1)))
        mon.end_episode(ep + 1, raw_return, success, steps, eps, env.curriculum_dist,
                        info.get("delivered", int(success)), info.get("n_passengers", 1),
                        getattr(env, "max_p", 1))

        if (ep + 1) % 100 == 0:
            w = np.array(log[-100:])
            extra = f" | entregues {w[:,6].sum() / max(1, w[:,7].sum()):.2f}" if args.env == "multi" else ""
            print(f"Ep {ep+1:5d} | retorno {w[:,1].mean():7.1f} | sucesso {w[:,2].mean():.2f}{extra} "
                  f"| passos {w[:,3].mean():6.1f} | eps {eps:.2f} | raio {env.curriculum_dist:3d} "
                  f"| {time.time()-t0:6.0f}s", flush=True)

        if (ep + 1) % args.eval_every == 0:
            stats, _ = evaluate(policy, args, device, args.eval_episodes, seed=args.eval_seed)
            print(f"   >> Avaliação DQN Essencial (mapas novos): {stats}", flush=True)
            score = (stats["success"], stats["delivered_frac"])
            is_best = score >= best_score
            if is_best:
                best_score = score
                torch.save(policy.state_dict(), f"dqn_taxi{tag}_best.pt")
            mon.log_eval(ep + 1, stats, is_best, args.loop_penalty)
            mon.save()
            mon.plot()

    torch.save(policy.state_dict(), f"dqn_taxi{tag}_final.pt")
    mon.finalize()
    print(f"Logs e gráficos gravados: train_log{tag}.csv, eval_log{tag}.csv, fig_treino{tag}.png, "
          f"fig_avaliacao{tag}.png")


def save_gif(frames, path):
    if not path or not frames:
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, duration=0.08)
        print("GIF salvo em", path)
    except ImportError:
        print("Instale imageio para salvar o GIF: pip install imageio")


if __name__ == "__main__":
    main()
