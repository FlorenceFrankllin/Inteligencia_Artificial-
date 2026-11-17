# Taxi DQN – Cidade 80x80 (GBC063 – Desafio 1)

Um táxi aprende, com DQN, a buscar de 1 a 3 passageiros num mapa 80x80 com obstáculos sorteados a cada
episódio. Cada passageiro nasce num lugar diferente e tem um destino diferente dos outros. Este guia é para
**Windows** e usa o **PowerShell**.

## Arquivos

| Arquivo | Função |
|---|---|
| train_dqn.py | Treino e avaliação do agente |
| viewer_cidade.py | Interface gráfica (cenário de cidade) |
| plot_results.py | Gera os gráficos a partir dos arquivos de log |
| monitor.py | Registra o treino e desenha os gráficos (usado pelos dois acima) |
| multi_taxi_env.py, big_taxi_env.py | Ambientes (1 a 3 passageiros / base) |
| safe_policy.py | Proteção contra o táxi ficar preso |
| viewer_pygame.py | Visualizador simples (também usado pelo viewer_cidade.py) |
| dados_exemplo/ | Exemplo de logs e gráficos de um treino real |

Mantenha **todos os .py na mesma pasta** e rode os comandos sempre dentro dela.

## 1. Instalação

**1.1 Python.** Instale o Python 3.10 ou mais novo em python.org. Na instalação, marque
**"Add python.exe to PATH"**. Confira: `py --version`

**1.2 Abrir o PowerShell na pasta do projeto.** Abra a pasta no Explorador de Arquivos, clique na barra de
endereço, digite `powershell` e aperte Enter.

**1.3 Criar e ativar o ambiente virtual** (isola as bibliotecas do projeto):

```
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Deve aparecer `(.venv)` no início da linha. Se o PowerShell reclamar de scripts desabilitados, rode
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e ative de novo. No Prompt de Comando (cmd), use
`.venv\Scripts\activate.bat`. **A cada terminal novo, ative o ambiente de novo.**

**1.4 Instalar as bibliotecas:**

```
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

(Com placa de vídeo NVIDIA, gere o comando do PyTorch em pytorch.org/get-started/locally. A CPU basta.)

**1.5 Testar a instalação:**

```
python -c "import torch, gymnasium, pygame, numpy, matplotlib; print('Tudo certo')"
```

## 2. Ver a interface sem treinar

```
python viewer_cidade.py --policy optimal --size 80
```

Mostra o táxi seguindo a rota ótima. Serve para confirmar que a janela abre. Feche com **Q** ou **Esc**.

## 3. Treinar o agente

Teste rápido (para ver se tudo roda e estimar a velocidade):

```
python train_dqn.py --size 20 --episodes 500
```

Treino recomendado:

```
python train_dqn.py --size 80 --episodes 4000 --double --dueling --memory --loop-penalty 0.3 --eval-every 250 --eval-episodes 30
```

- Demora: de dezenas de minutos a várias horas, conforme o computador. Em uma máquina de referência,
  2000 episódios levaram ~22 min e 4000 episódios ~1h30.
- Pode parar a qualquer momento com **Ctrl+C**: o melhor modelo e os logs já estão salvos.
- A cada avaliação, a linha `>> Avaliação` mostra `success` (episódios completos), `delivered_frac`
  (passageiros entregues) e `steps_vs_optimal` (1.0 = rota ótima).
- Arquivos gerados na pasta: `dqn_taxi_multi_best.pt` (melhor modelo, **use este**), `dqn_taxi_multi_final.pt`,
  `train_log_multi.csv`, `eval_log_multi.csv`, `fig_treino_multi.png`, `fig_avaliacao_multi.png`.
- Rodar o treino de novo **sobrescreve** esses arquivos: copie antes o que quiser guardar.

## 4. Executar o agente treinado (interface gráfica)

```
python viewer_cidade.py --model dqn_taxi_multi_best.pt --size 80
```

O programa descobre sozinho o tipo de modelo. Teclas:

| Tecla | Ação |
|---|---|
| Espaço | Pausa / continua |
| Seta direita | Um passo (em pausa) |
| Setas cima / baixo | Velocidade |
| C | Alterna: câmera que segue o táxi / mapa inteiro |
| + e - | Zoom |
| R | Novo episódio |
| T | Mostra/esconde o rastro |
| V | Mostra a área que a rede enxerga |
| Q ou Esc | Sai |

Opções úteis: `--policy optimal` (rota ótima, para comparar), `--max-passengers 2` (limita os passageiros),
`--vp 700` (janela menor, para telas pequenas), `--loop-penalty 0` (desliga só a penalidade de revisita).

## 5. Gerar os gráficos para a apresentação

**5.1 Resultados finais por número de passageiros** (50 mapas novos cada; com e sem a penalidade de revisita).
Cada linha é gravada em `results.csv`:

```
foreach ($p in 1,2,3) { foreach ($lp in '0.3','0') { python train_dqn.py --size 80 --eval-only dqn_taxi_multi_best.pt --loop-penalty $lp --min-passengers $p --max-passengers $p --save-results results.csv } }
```

**5.2 Gerar as figuras** (lê `train_log_multi.csv`, `eval_log_multi.csv` e `results.csv` da pasta):

```
python plot_results.py --tag _multi
```

Criados: `fig_treino_multi.png` (slide de treinamento), `fig_final_multi.png` (slide de resultados) e
`fig_avaliacao_multi.png`. Opções: `--out-dir figuras` (outra pasta de saída), `--window 50` (suavização).

**Exemplo com dados prontos** (não precisa treinar):

```
python plot_results.py --dir dados_exemplo --train dados_exemplo\train_log_multi_resumo.csv --tag _multi --window 1 --out-dir figuras
```

## 6. Problemas comuns

| Problema | O que fazer |
|---|---|
| `python` não é reconhecido | Use `py` no lugar de `python`, ou reinstale o Python marcando "Add to PATH" |
| `ModuleNotFoundError` (torch, pygame...) | Ative o ambiente (passo 1.3) e repita o 1.4 |
| `ModuleNotFoundError: big_taxi_env` | Rode dentro da pasta, com todos os .py juntos |
| `FileNotFoundError: dqn_taxi_multi_best.pt` | Treine primeiro ou coloque o arquivo .pt na pasta |
| `O modelo espera entrada de tamanho...` | Use o mesmo `--view` do treino (padrão 11) |
| A janela não cabe na tela | Use `--vp 700` |
| Treino parece parado | Cada bloco de 100 episódios demora; espere a próxima linha |

## Opcional: mapa de ruas (o táxi só anda pelas ruas)

```
python viewer_cidade.py --policy optimal --map streets
python train_dqn.py --map streets --size 80 --episodes 4000 --double --dueling --memory --loop-penalty 0.3
python viewer_cidade.py --model dqn_taxi_multi_streets_best.pt --map streets
python plot_results.py --tag _multi_streets
```

Neste modo é preciso treinar um modelo próprio (o modelo do mapa aberto não foi feito para ruas).
