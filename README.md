# Inteligência Artificial  | 

Repositório dedicado ao desenvolvimento prático e teórico da disciplina de **Inteligência Artificial** .

* **Docente:** Prof. Marcelo Keese Albertini[cite: 1]
* **Discentes:**
  * Florence Franklin Jeronimo[cite: 253]
  * Luiz Fellipe Machado Tristão[cite: 253]

---

## 📌 Visão Geral da Disciplina

O curso aborda a inteligência artificial sob a perspectiva de **tomada de decisão sequencial sob incerteza**, com ênfase aprofundada em **Aprendizagem por Reforço (Reinforcement Learning - RL)** (baseada na estrutura do CS234 de Stanford)[cite: 1, 4, 17, 21]. 

Ao longo do período, os tópicos estudados compreendem:
1. **Fundamentos e Planejamento com Modelo:** Processos de Markov, Processos de Recompensa de Markov (MRPs) e Processos de Decisão de Markov (MDPs); Equação de Bellman e Programação Dinâmica (Iteração de Política e Iteração de Valor)[cite: 2, 33, 41, 55, 81, 90].
2. **Avaliação e Predição Sem Modelo:** Estimação do valor $V^\pi$ via Monte Carlo (MC), Diferenças Temporais ($TD(0)$) e Equivalência de Certeza em regimes de fluxo e em lote (*batch*)[cite: 117, 121, 146, 153, 181].
3. **Controle Tabular Sem Modelo:** Dilema de exploração vs. aproveitamento ($\epsilon$-guloso e condição GLIE), algoritmos SARSA (*on-policy*) e Q-learning (*off-policy*), análise do viés de maximização e Double Q-learning[cite: 176, 177, 201, 203, 204, 205, 211, 232].
4. **Aproximação de Função Valor (VFA) e Deep RL:** Transição do formato tabular para aproximadores paramétricos, a "Tríade Mortal" (VFA + bootstrapping + off-policy) e arquiteturas Deep Q-Networks (DQN) com *experience replay* e *target networks* fixas[cite: 212, 214, 221, 223, 224, 225].
5. **Métodos Baseados em Política e Tópicos Avançados:** Teorema do Gradiente de Política, arquiteturas Ator-Crítico, otimização de políticas proximais (PPO) e conexões com alinhamento de modelos de linguagem (RLHF/DPO)[cite: 5, 17, 234].

---

## 🚕 Sobre o Projeto Prático

O projeto da disciplina é desenvolvido de maneira incremental ao longo do semestre, acompanhando os desafios práticos propostos pelo professor. O objetivo central é projetar, implementar, avaliar e comparar agentes autônomos de aprendizagem por reforço aplicados ao domínio do **Taxi**.

O desenvolvimento ocorre em etapas cumulativas:
* **Fase 1 (Desafio 1):** Modelagem formal do ambiente como um Processo de Decisão de Markov (definição rigorosa de estados markovianos, ações e função de recompensa)[cite: 24, 65].
* **Fase 2:** Implementação e benchmarking de métodos tabulares clássicos sem modelo (SARSA vs. Q-learning) avaliando eficiência amostral e estabilidade de convergência[cite: 204, 205, 206].
* **Fase 3:** Transição para espaços contínuos/vetorizados e implementação de controle com aproximação de função e Deep Q-Networks (DQN)[cite: 213, 220, 226].
* **Fases Seguintes:** Otimização de políticas, refinamento de hiperparâmetros e análise empírica comparativa dos resultados.

---

## 📂 Organização do Repositório

```text
├── docs/                 # Notas técnicas, deduções matemáticas e referências
├── presentation/         # Apresentações e slides dos marcos do projeto
├── desafio-1/            # Entrega 1: Modelagem matemática do MDP (Taxi)
│   ├── README.md         # Detalhamento específico do Desafio 1
│   └── slides.pdf        # Apresentação do Desafio 1
└── src/                  # Código-fonte das simulações, ambientes e algoritmos
