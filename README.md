# Inteligência Artificial 

Repositório destinado ao envio e acompanhamento do projeto com atividades práticas  da disciplina de **Inteligência Artificial ** 

* **Docente:** Prof. Marcelo Keese Albertini
* **Discentes:**
  * Florence Franklin Jeronimo
  * Luiz Fellipe Machado Tristão
* **Domínio de Estudo:** Ambiente *Taxi*

---

## Desafio 1: Modelagem do Processo de Decisão de Markov (MDP)

O primeiro desafio consiste na formalização teórica do ambiente do **Taxi** segundo a tupla de decisão sequencial sob incerteza $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

### 1. Espaço de Estados ($\mathcal{S}$) e Propriedade de Markov
O ambiente é totalmente observável e o estado do sistema é expresso de forma discreta:
* **Posição do Táxi:** Coordenadas cartesianas em grade $5 \times 5$ ($25$ posições discretas).
* **Posição do Passageiro:** $5$ possibilidades ($4$ pontos cardeais de embarque $R, G, Y, B$ ou dentro do táxi).
* **Destino do Passageiro:** $4$ locais possíveis ($R, G, Y, B$).
* **Cardinalidade do Espaço:** $\vert{}\mathcal{S}\vert{} = 25 \times 5 \times 4 = 500$ estados possíveis.

> **Validação de Markov:** O estado $s_t = (\text{linha}, \text{coluna}, \text{passageiro}, \text{destino})$ atende rigorosamente a condição $\mathbb{P}(s_{t+1} \mid s_t, a_t) = \mathbb{P}(s_{t+1} \mid h_t, a_t)$, funcionando como estatística suficiente do histórico. O movimento futuro depende exclusivamente da posição e do status atual, sem dependência de trajetórias passadas.

### 2. Espaço de Ações ($\mathcal{A}$)
O agente possui 6 ações discretas ($\vert{}\mathcal{A}\vert{} = 6$):
* `0`: Mover Sul ($\downarrow$)
* `1`: Mover Norte ($\uparrow$)
* `2`: Mover Leste ($\rightarrow$)
* `3`: Mover Oeste ($\leftarrow$)
* `4`: *Pickup* (Apanhar passageiro)
* `5`: *Dropoff* (Desembarcar passageiro)

*Dinâmica de transição:* Transições desimpedidas são determinísticas. Colisões com barreiras ou bordas mantêm o agente na mesma célula. Ações de embarque/desembarque fora de locais válidos não alteram o estado físico do passageiro.

### 3. Função de Recompensa ($\mathcal{R}$) e Regime da Tarefa
A função $R(s, a, s')$ incentiva a otimização temporal e pune erros operacionais:
* $+20$: Conclusão da viagem com entrega bem-sucedida no destino correto (estado terminal).
* $-10$: Execução de *Pickup* ou *Dropoff* ilegal (fora do local correto).
* $-1$: Custo a cada passo executado (penalidade temporal que força o agente a encontrar a rota geodésica mínima).

* **Regime:** Tarefa episódica com limite de passos.
* **Fator de Desconto:** $\gamma = 0{,}99$ (horizonte efetivo $\approx 100$ passos).

---

## Estrutura do Repositório

```text
├── docs/                 # Documentação complementar e notas de estudo
├── presentation/         # Slides de apresentação do Desafio 1 (PDF)
├── src/                  # Implementação do ambiente e algoritmos (Q-Learning / DQN)
└── README.md             # Síntese do projeto e instruções de execução
