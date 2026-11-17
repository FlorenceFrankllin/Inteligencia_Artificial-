"""
plot_results.py
Gera as figuras para os slides a partir dos CSVs (não precisa de PyTorch nem de GPU).

Exemplos:
  python plot_results.py --tag _multi                       # lê train_log_multi.csv, eval_log_multi.csv, results.csv
  python plot_results.py --tag _multi --out-dir figuras     # grava os PNGs em outra pasta
  python plot_results.py --train train_log_multi_resumo.csv --eval eval_log_multi.csv --tag _multi

Figuras geradas (as que tiverem dados):
  fig_treino{tag}.png      retorno, sucesso/entregues, passos, epsilon e currículo (slide de treinamento)
  fig_avaliacao{tag}.png   avaliação periódica em mapas novos
  fig_final{tag}.png       resultado final por número de passageiros (+ ablação da proteção)
  fig_diagnostico{tag}.png perda e Q médio (só em logs gerados pela versão nova do treino)
"""
import argparse

from monitor import plot_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="pasta com os CSVs")
    p.add_argument("--tag", default="", help='sufixo dos arquivos, ex.: "_multi" ou "_multi_streets"')
    p.add_argument("--train", default=None, help="caminho do log de treino (padrão: train_log{tag}.csv)")
    p.add_argument("--eval", default=None, help="caminho do log de avaliação (padrão: eval_log{tag}.csv)")
    p.add_argument("--results", default="results.csv", help="resultados finais (de --save-results)")
    p.add_argument("--window", type=int, default=100, help="janela da média móvel (episódios)")
    p.add_argument("--best-ep", type=int, default=None, help="episódio do melhor modelo (padrão: do eval_log)")
    p.add_argument("--out-dir", default=None, help="pasta de saída (padrão: a mesma dos CSVs)")
    a = p.parse_args()
    made = plot_all(a.dir, a.tag, a.results, a.window, a.best_ep, a.out_dir, a.train, a.eval)
    if not made:
        print("Nenhum CSV encontrado. Rode o treino primeiro ou aponte --train / --eval / --results.")
    for m in made:
        print("gerado:", m)


if __name__ == "__main__":
    main()
