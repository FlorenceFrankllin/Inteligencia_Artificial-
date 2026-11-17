"""
monitor.py
Monitoramento do treino e geração dos gráficos para os slides.

Duas partes:
  1) TrainingMonitor: usado pelo train_dqn.py. Registra, a cada episódio, retorno, sucesso, passos, epsilon,
     passageiros entregues, perda (loss) e Q médio; registra cada avaliação periódica; grava CSVs
     (train_log*.csv e eval_log*.csv) e atualiza as figuras a cada avaliação, então nada se perde
     se o treino for interrompido com Ctrl+C.
  2) Funções de gráfico (plot_training, plot_eval, plot_final, plot_diagnostics): leem os CSVs, então
     funcionam sem PyTorch e podem ser rodadas depois, pelo plot_results.py.

Arquivos de entrada/saída (tag = "", "_multi", "_multi_streets", ...):
  train_log{tag}.csv   um registro por episódio            -> fig_treino{tag}.png, fig_diagnostico{tag}.png
  eval_log{tag}.csv    um registro por avaliação periódica -> fig_avaliacao{tag}.png
  results.csv          resultados finais (--eval-only --save-results results.csv) -> fig_final.png
"""
import csv
import os

import numpy as np

AZUL, MARINHO, LARANJA, VERDE, VERMELHO, CINZA = "#3373B6", "#0B3B6B", "#E8832A", "#2E9E5B", "#D64545", "#7A8594"

TRAIN_COLS = ["episode", "raw_return", "success", "steps", "epsilon", "curriculum_radius", "delivered",
              "n_passengers", "max_passengers", "loss", "q_mean"]
EVAL_COLS = ["episode", "return", "success", "delivered_frac", "steps_vs_optimal", "is_best", "loop_penalty"]
RESULT_COLS = ["model", "variant", "passengers", "loop_penalty", "map", "episodes", "seed",
               "success", "delivered_frac", "steps_vs_optimal", "return"]


# ============================================================ arquivos CSV
def append_result(path, row):
    """Acrescenta uma linha de resultado final em results.csv (cria o arquivo se não existir)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_COLS})


def read_csv(path):
    """Devolve (colunas numéricas como arrays, linhas cruas). Texto vira NaN nas colunas numéricas."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in (rows[0].keys() if rows else []):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        cols[k] = np.array(vals)
    return cols, rows


def moving_average(y, window):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - window + 1): i + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmean(seg)
    return out


def _rolling_ratio(num, den, window):
    num, den = np.asarray(num, float), np.asarray(den, float)
    out = np.full(len(num), np.nan)
    for i in range(len(num)):
        lo = max(0, i - window + 1)
        d = den[lo: i + 1].sum()
        out[i] = num[lo: i + 1].sum() / d if d > 0 else np.nan
    return out


def load_training(path, window=100):
    """Lê train_log e devolve séries prontas para plotar.
    Aceita o log por episódio (suaviza com média móvel) ou um log já agregado
    (com a coluna delivered_frac, por exemplo um resumo a cada 100 episódios)."""
    cols, _ = read_csv(path)
    ep = cols["episode"]
    if "delivered_frac" in cols:  # já agregado
        d = dict(ep=ep, ret=cols["raw_return"], succ=cols["success"], deliv=cols["delivered_frac"],
                 steps=cols["steps"], eps=cols["epsilon"])
        d["maxp"] = cols.get("max_passengers")
        d["loss"], d["q"] = cols.get("loss"), cols.get("q_mean")
        return d
    d = dict(ep=ep, ret=moving_average(cols["raw_return"], window), succ=moving_average(cols["success"], window),
             steps=moving_average(cols["steps"], window), eps=cols["epsilon"])
    d["deliv"] = (_rolling_ratio(cols["delivered"], cols["n_passengers"], window)
                  if "delivered" in cols else d["succ"])
    if "max_passengers" in cols and np.isfinite(cols["max_passengers"]).any():
        d["maxp"] = cols["max_passengers"]
    elif "n_passengers" in cols:
        d["maxp"] = np.maximum.accumulate(cols["n_passengers"])
    else:
        d["maxp"] = None
    d["loss"] = moving_average(cols["loss"], window) if "loss" in cols else None
    d["q"] = moving_average(cols["q_mean"], window) if "q_mean" in cols else None
    return d


def load_eval(path):
    cols, _ = read_csv(path)
    best = None
    if "is_best" in cols:
        idx = np.flatnonzero(cols["is_best"] == 1)
        best = int(cols["episode"][idx[-1]]) if len(idx) else None
    pen = float(cols["loop_penalty"][0]) if "loop_penalty" in cols and len(cols["loop_penalty"]) else 0.0
    return dict(ep=cols["episode"], ret=cols["return"], succ=cols["success"], deliv=cols["delivered_frac"],
                ratio=cols["steps_vs_optimal"], best_ep=best, penalty=pen)


# ================================================================ gráficos
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
                         "axes.titlesize": 14, "legend.frameon": False})
    return plt


def _mark_best(ax, best_ep, ytext=0.97):
    if best_ep:
        ax.axvline(best_ep, ls="--", color=LARANJA, lw=1.6)
        ax.text(best_ep, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * ytext,
                f" melhor modelo\n (ep. {best_ep})", color=LARANJA, va="top", fontsize=11)


def plot_training(train_csv, eval_csv=None, out="fig_treino.png", window=100, best_ep=None, title=None):
    """Figura 2x2: retorno, sucesso/entregues (com a avaliação em mapas novos), passos, epsilon e currículo."""
    plt = _plt()
    d = load_training(train_csv, window)
    ev = load_eval(eval_csv) if eval_csv and os.path.exists(eval_csv) else None
    if best_ep is None and ev is not None:
        best_ep = ev["best_ep"]
    ep = d["ep"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))

    ax = axs[0, 0]
    ax.plot(ep, d["ret"], color=AZUL, lw=2)
    ax.set_title("Retorno médio por episódio")
    ax.set_ylabel("retorno")
    _mark_best(ax, best_ep, 0.35)

    ax = axs[0, 1]
    ax.plot(ep, d["succ"], color=VERDE, lw=2, label="episódios completos (treino)")
    ax.plot(ep, d["deliv"], color=AZUL, lw=2, ls="--", label="passageiros entregues (treino)")
    if ev is not None:
        ax.plot(ev["ep"], ev["succ"], "o:", color=LARANJA, ms=6, lw=1.2, label="avaliação (mapas novos)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Sucesso e passageiros entregues")
    ax.set_ylabel("fração")
    ax.legend(loc="upper right", fontsize=10)
    _mark_best(ax, best_ep, 0.27)

    ax = axs[1, 0]
    ax.plot(ep, d["steps"], color=MARINHO, lw=2)
    top = np.nanmax(d["steps"])
    if top >= 590:
        lim = 1200 if top > 700 else 600
        ax.axhline(lim, ls=":", color=CINZA)
        ax.text(ep[0], lim, " limite de passos", color=CINZA, va="bottom", fontsize=11)
    ax.set_title("Passos por episódio")
    ax.set_ylabel("passos")
    ax.set_xlabel("episódios")
    _mark_best(ax, best_ep, 0.35)

    ax = axs[1, 1]
    ax.plot(ep, d["eps"], color=LARANJA, lw=2, label="ε (exploração)")
    ax.set_title("Exploração e currículo")
    ax.set_ylabel("ε")
    ax.set_xlabel("episódios")
    if d["maxp"] is not None and np.isfinite(d["maxp"]).any():
        ax2 = ax.twinx()
        ax2.step(ep, d["maxp"], where="pre", color=CINZA, lw=2)
        ax2.set_ylabel("máx. de passageiros sorteados", color=CINZA)
        ax2.set_yticks([1, 2, 3])
        ax2.set_ylim(0.5, 3.5)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_eval(eval_csv, out="fig_avaliacao.png", title=None):
    """Avaliação periódica em mapas novos: sucesso, passageiros entregues e passos/ótimo."""
    plt = _plt()
    ev = load_eval(eval_csv)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    ax.plot(ev["ep"], ev["succ"], "o-", color=VERDE, lw=2, label="episódios completos")
    ax.plot(ev["ep"], ev["deliv"], "s--", color=AZUL, lw=2, label="passageiros entregues")
    ax.set_ylim(-0.03, 1.08)
    how = f"com proteção {ev['penalty']:g}".replace(".", ",") if ev["penalty"] > 0 else "só máscara de ações"
    ax.set_title(f"Avaliação em mapas novos ({how})")
    ax.set_xlabel("episódios de treino")
    ax.set_ylabel("fração")
    ax.legend(loc="upper right")
    if ev["best_ep"]:
        i = int(np.flatnonzero(ev["ep"] == ev["best_ep"])[0])
        ax.plot([ev["ep"][i]], [ev["succ"][i]], "*", color=LARANJA, ms=20, zorder=5)
        ax.annotate(f"melhor modelo\n(ep. {ev['best_ep']})", (ev["ep"][i], ev["succ"][i]),
                    textcoords="offset points", xytext=(12, -34), color=LARANJA, fontsize=11)
    ax = axs[1]
    ok = np.isfinite(ev["ratio"])
    ax.plot(ev["ep"][ok], ev["ratio"][ok], "o-", color=MARINHO, lw=2)
    ax.axhline(1.0, ls=":", color=CINZA)
    ax.text(ev["ep"][0], 1.0, " ótimo = 1,0", color=CINZA, va="bottom", fontsize=11)
    ax.set_title("Passos gastos / passos ótimos (episódios concluídos)")
    ax.set_xlabel("episódios de treino")
    ax.set_ylabel("razão")
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_diagnostics(train_csv, out="fig_diagnostico.png", window=100):
    """Perda (Huber) e Q médio do lote: ajudam a diagnosticar divergência. Só existe em logs novos."""
    d = load_training(train_csv, window)
    if d["loss"] is None or not np.isfinite(d["loss"]).any():
        return None
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].plot(d["ep"], d["loss"], color=VERMELHO, lw=2)
    axs[0].set_yscale("log")
    axs[0].set_title("Perda do DQN (Huber)")
    axs[0].set_xlabel("episódios")
    axs[1].plot(d["ep"], d["q"], color=MARINHO, lw=2)
    axs[1].set_title("Q médio das ações tomadas no lote")
    axs[1].set_xlabel("episódios")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_final(results_csv, out="fig_final.png", title=None):
    """Barras por número de passageiros: sucesso (%) e passos/ótimo, uma série por (modelo, proteção)."""
    plt = _plt()
    _, rows = read_csv(results_csv)
    names = {"best": "melhor modelo", "final": "modelo final"}
    series = {}
    for r in rows:
        key = f"{names.get(r['model'], r['model'])} · {r['variant']}"
        series.setdefault(key, {})[r["passengers"]] = r
    groups = sorted({r["passengers"] for r in rows})
    palette = [AZUL, LARANJA, CINZA, VERDE, VERMELHO]
    pt = lambda txt: str(txt).replace(".", ",")          # vírgula decimal (padrão pt-BR) nos rótulos
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    width = 0.8 / max(1, len(series))
    handles = {}
    for gi, g in enumerate(groups):
        present = [(k, label, byp[g]) for k, (label, byp) in enumerate(series.items()) if g in byp]
        for j, (k, label, r) in enumerate(present):
            x = gi + (j - (len(present) - 1) / 2) * width
            succ = float(r["success"]) * 100
            h = axs[0].bar(x, succ, width * 0.92, color=palette[k % 5])
            handles.setdefault(label, h)
            axs[0].text(x, succ + 1.5, f"{succ:.0f}%", ha="center", fontsize=11)
            ratio = float(r["steps_vs_optimal"]) if r["steps_vs_optimal"] not in ("", "nan") else np.nan
            if np.isfinite(ratio) and succ > 0:
                axs[1].bar(x, ratio, width * 0.92, color=palette[k % 5])
                axs[1].text(x, ratio + 0.01, pt(f"{ratio:.2f}"), ha="center", fontsize=11)
    for ax in axs:
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g} passageiro" + ("" if g == "1" else "s") for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
    axs[0].set_ylim(0, 112)
    axs[0].set_ylabel("episódios completos (%)")
    axs[0].set_title("Taxa de sucesso (mapas novos)")
    axs[1].set_ylim(0.9, max(1.4, axs[1].get_ylim()[1]))
    axs[1].axhline(1.0, ls=":", color=CINZA)
    axs[1].set_ylabel("passos / passos ótimos")
    axs[1].set_title("Eficiência nos episódios concluídos (1,0 = ótimo)")
    fig.legend(list(handles.values()), [pt(l) for l in handles], loc="lower center", ncol=len(handles),
               fontsize=12)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_all(directory=".", tag="", results="results.csv", window=100, best_ep=None, out_dir=None,
             train_csv=None, eval_csv=None):
    """Gera todas as figuras possíveis com os arquivos que existirem. Devolve a lista de PNGs criados."""
    out_dir = out_dir or directory
    os.makedirs(out_dir, exist_ok=True)
    tr = train_csv or os.path.join(directory, f"train_log{tag}.csv")
    ev = eval_csv or os.path.join(directory, f"eval_log{tag}.csv")
    rs = results if os.path.isabs(results) or os.path.exists(results) else os.path.join(directory, results)
    made = []
    if os.path.exists(tr):
        made.append(plot_training(tr, ev if os.path.exists(ev) else None,
                                  os.path.join(out_dir, f"fig_treino{tag}.png"), window, best_ep))
        diag = plot_diagnostics(tr, os.path.join(out_dir, f"fig_diagnostico{tag}.png"), window)
        if diag:
            made.append(diag)
    if os.path.exists(ev):
        made.append(plot_eval(ev, os.path.join(out_dir, f"fig_avaliacao{tag}.png")))
    if os.path.exists(rs):
        made.append(plot_final(rs, os.path.join(out_dir, f"fig_final{tag}.png")))
    return made


# ========================================================= monitor do treino
class TrainingMonitor:
    """Registra o treino episódio a episódio e mantém CSVs e figuras sempre atualizados."""

    def __init__(self, tag="", out_dir=".", window=100):
        self.tag, self.out_dir, self.window = tag, out_dir, window
        self.train_rows, self.eval_rows = [], []
        self._loss, self._q = [], []
        self._finalized, self._warned = False, False
        self.train_csv = os.path.join(out_dir, f"train_log{tag}.csv")
        self.eval_csv = os.path.join(out_dir, f"eval_log{tag}.csv")

    def record_update(self, loss, q_mean):
        """Chame a cada passo de otimização com a perda e o Q médio do lote."""
        self._loss.append(loss)
        self._q.append(q_mean)

    def end_episode(self, ep, raw_return, success, steps, eps, radius, delivered, n_passengers, max_passengers):
        loss = float(np.mean(self._loss)) if self._loss else float("nan")
        q = float(np.mean(self._q)) if self._q else float("nan")
        self._loss, self._q = [], []
        self.train_rows.append([ep, raw_return, float(success), steps, eps, radius, delivered,
                                n_passengers, max_passengers, loss, q])

    def log_eval(self, ep, stats, is_best, loop_penalty=0.0):
        self.eval_rows.append([ep, stats["return"], stats["success"], stats["delivered_frac"],
                               stats["steps_vs_optimal"], int(is_best), loop_penalty])

    def save(self):
        for path, cols, rows in ((self.train_csv, TRAIN_COLS, self.train_rows),
                                 (self.eval_csv, EVAL_COLS, self.eval_rows)):
            if not rows:
                continue
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)

    def plot(self):
        """Atualiza as figuras (nunca derruba o treino se o matplotlib faltar)."""
        try:
            plot_all(self.out_dir, self.tag, window=self.window, results="__nenhum__.csv")
        except ImportError:
            if not self._warned:
                print("(matplotlib não instalado: pulei os gráficos; os CSVs foram salvos)")
                self._warned = True
        except Exception as e:  # gráfico nunca deve interromper o treino
            if not self._warned:
                print(f"(aviso: não consegui gerar os gráficos: {e})")
                self._warned = True

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        self.save()
        self.plot()
