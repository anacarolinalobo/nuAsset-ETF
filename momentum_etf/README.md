# Índice de Momentum Long-Only — Ações Brasileiras

Entregável de código para o business case de Quantitative Specialist
(Nu Asset Management). Implementa a metodologia proposta para um índice de
momentum long-only sobre ações brasileiras, o backtest 2010–2026, as
métricas de desempenho pedidas e análises complementares de robustez.

## Aviso importante sobre os dados

Os três CSVs do case (`acoes_retornos.csv`, `ibov_composicao.csv`,
`benchmarks_diarios.csv`) **não foram anexados a esta sessão** — este
ambiente não tem acesso a eles nem à internet para buscar fontes
alternativas de mercado. Por isso:

- Todo o código foi escrito contra o **schema descrito no briefing** e
  está pronto para rodar assim que os arquivos reais forem colocados em
  `data/`.
- `scripts/make_sample_data.py` gera dados **sintéticos** (aleatórios, com
  uma tendência injetada artificialmente) só para exercitar o pipeline de
  ponta a ponta neste ambiente — os números do `output/dashboard.html`
  gerado a partir deles **não têm nenhum significado econômico** e não
  devem ser usados para avaliar a estratégia. Rode
  `python scripts/run_backtest.py` novamente depois de colocar os CSVs
  reais em `data/` para obter os resultados de verdade.
- Nenhuma fonte externa foi incorporada à análise (não usei nenhum dado
  complementar de B3, CVM, ou provedores de mercado) — é uma limitação
  desta entrega, não uma escolha metodológica, e está listada abaixo em
  "O que ficou de fora".

## Como rodar

```bash
pip install -r requirements.txt

# 1) smoke test com dados sintéticos (opcional, só para validar o pipeline)
python scripts/make_sample_data.py

# 2) com os CSVs reais em data/ (substituindo os sintéticos, se gerados)
python scripts/run_backtest.py

# 3) validação treino/teste da grade de parâmetros (checagem de overfitting,
#    pode levar vários minutos — 1 backtest por combinação da grade)
python scripts/run_train_test_split.py

# 4) testes unitários
pytest -q
```

Saídas em `output/`: `dashboard.html` (relatório visual), `summary_metrics.csv`,
`daily_returns.csv`, `turnover_history.csv`, `attribution.csv`,
`sensitivity_lookback.csv`, `sensitivity_cost.csv` e, com
`data/liquidez_mercado.csv`, `liquidez_carteira.csv`,
`capacidade_ultima_carteira.csv`, `cobertura_liquidez.csv`. Saídas de
`scripts/run_train_test_split.py` em `output/train_test_split/`:
`grid_treino_teste.csv`, `resumo.csv`.

## Estrutura do código

```
src/
  config.py            parâmetros da metodologia, centralizados
  data_loader.py        carga e limpeza dos 3 CSVs; checagem de dado suspeito
  market_data.py        volume (B3) e market cap (CVM): ADTV, filtros, sem look-ahead
  universe.py            universo elegível (entrada/saída, sem look-ahead)
  signal.py               sinal de momentum (12-1, ajustado a risco, z-score)
  selection.py            seleção com banda de turnover (buffer rule)
  weighting.py            ponderação por score, com teto por ativo
  rebalance.py            calendário trimestral + defasagem de execução
  corporate_actions.py    tratamento de delisting
  costs.py                custo de transação por turnover
  backtest.py              motor do backtest (loop diário)
  metrics.py               métricas de desempenho
  attribution.py           atribuição de performance vs. Ibovespa
  sensitivity.py           sensibilidade a parâmetros / estabilidade temporal
  capacity.py              estimativa de capacidade do produto
  report.py                dashboard HTML autocontido
scripts/
  run_backtest.py          orquestra tudo, ponta a ponta
  run_train_test_split.py  validação treino/teste da grade de parâmetros
  make_sample_data.py      gera dados sintéticos (smoke test apenas)
tests/                     testes unitários (pytest) com dados sintéticos pequenos
```

Cada módulo faz uma coisa (carregar dado, medir sinal, selecionar, pesar,
simular, medir performance), o que deixa o `backtest.py` como o único
lugar que precisa saber a ORDEM das operações — trocar, por exemplo, o
critério de ponderação não exige tocar em `selection.py` nem em
`backtest.py`.

## 1. Metodologia do índice

### Universo elegível

Ação entra no universo elegível em uma data de rebalanceamento se, **usando
apenas dados até aquela data**:

1. Tem pelo menos `lookback_days + skip_days` (padrão: 273) pregões de
   histórico — para que o sinal de 12 meses esteja plenamente formado (sem
   isso o sinal seria calculado sobre uma janela incompleta, distorcendo o
   score).
2. Ainda está "viva": teve ao menos um retorno observado nos últimos
   `liquidity_window_days` (126) pregões — se não, presume-se deslistada.
3. Passa em um proxy de liquidez: fração de pregões com retorno observado
   na mesma janela >= `min_active_ratio` (90%).

**Com dados de liquidez, volume e market cap** (`data/liquidez_mercado.csv`,
gerado pela última célula de `caseNuAsset.ipynb` a partir do COTAHIST da B3
e do capital social da CVM — ver `src/market_data.py`), o critério 3 passa a
usar dado real de negociação em vez do proxy:

3a. Fração de pregões com volume > 0 em `liquidity_window_days` >= `min_active_ratio`.
3b. ADTV (mediana do volume financeiro diário em `adtv_window_days` = 63
    pregões) >= `min_adtv_brl` (R$ 5 mi).
4.  Market cap mais recente conhecido até a data >= `min_market_cap_brl`
    (R$ 500 mi). Papel sem market cap na base passa por padrão
    (`require_market_cap=False`), porque a cobertura CVM x B3 ainda é parcial.

Sem esse arquivo o pipeline roda exatamente como antes (proxy).

A ação sai do universo automaticamente quando seu histórico de retorno
acaba (sem regra explícita de remoção — a ausência de dado já resolve
isso) ou quando deixa de passar no filtro de liquidez.

**Por que isso evita survivorship bias:** como `acoes_retornos.csv` inclui
ações que saíram de negociação até o último dia em que negociaram, o
universo em cada data histórica é calculado exatamente como teria sido
calculado *naquele momento* — nada é removido retroativamente por "ter
morrido depois".

### Definição do sinal de momentum

Momentum "12-1" ajustado a risco, clássico da literatura (Jegadeesh &
Titman, 1993; a variante risk-adjusted segue o espírito de Barroso &
Santa-Clara, 2015, sobre "momentum crashes"):

```
ret_acum(t)  = retorno acumulado de (t - 252) até (t - 21) pregões
vol(t)       = desvio-padrão diário dos retornos na mesma janela de 252 dias
score_bruto  = ret_acum(t) / max(vol(t), piso_vol)
score(t)     = z-score de score_bruto entre os elegíveis na data t
```

- **Por que pular os últimos 21 dias (~1 mês):** o retorno do último mês
  tende a reverter no curtíssimo prazo (efeito de microestrutura / mean
  reversion de curto prazo), um padrão distinto — e historicamente oposto
  — ao efeito de continuação que o momentum de médio prazo captura. Sem o
  skip, o sinal mistura dois efeitos com sinal esperado contrário.
- **Por que ajustar a risco:** um retorno acumulado de 40% em uma ação com
  vol diária de 6% é muito menos "sinal de tendência" que os mesmos 40% em
  uma ação com vol de 1.5%. Dividir pela vol favorece tendências mais
  consistentes e penaliza ações que só subiram por serem voláteis —
  reduz também a concentração do índice em nomes de baixa liquidez, que
  tendem a ter vol mais alta e ruidosa.

### Critério de seleção

Regra de banda ("buffer rule"), como na metodologia MSCI Momentum:

- Papel **fora** da carteira só entra se `score >= percentil 70` do
  universo elegível (`entry_percentile`).
- Papel **já na** carteira só sai se `score < percentil 60`
  (`hold_percentile`) — banda mais larga.
- Limites de tamanho: entre `min_names` (30) e `max_names` (60) ativos.

**Por que a banda, e não um corte único:** sem ela, um papel cujo score
oscila em torno do corte entra e sai a cada rebalanceamento só por ruído
estatístico, gerando turnover sem ganho de retorno esperado — a banda cria
histerese: para trocar de estado, o sinal precisa se mover o suficiente
para não ser explicável por ruído de curto prazo.

### Critério de ponderação

Peso proporcional ao score de momentum (truncado em zero, então papéis com
score negativo dentro da carteira — o que só acontece perto do corte de
`hold_percentile` — recebem peso zero e são efetivamente removidos no
próximo passo), com **teto de 8% por ativo** (`weight_cap`), redistribuído
proporcionalmente entre os demais.

**Extensões com market cap e volume** (desligadas por padrão, em
`MomentumConfig`):
- `weighting_scheme="score_sqrt_mcap"`: peso ∝ score × √market cap —
  inclina para nomes maiores sem virar cap-weight.
- `target_aum_brl`: teto de peso por papel derivado do ADTV
  (`ADTV × max_adtv_participation × days_to_build_position / AUM`). O que
  não couber em nenhum papel fica em caixa (CDI).

**Alternativas descartadas:**
- *Peso igual*: mais simples e menos concentrado, mas dilui o tilt de
  momentum — um papel com sinal apenas ligeiramente acima do corte pesaria
  o mesmo que o de maior sinal, o que vai contra o racional de comprar mais
  convicção onde o sinal é mais forte.
- *Peso por valor de mercado*: replicaria a distorção de cap-weight que um
  índice de momentum busca evitar, e a base fornecida não traz valor de
  mercado/free float diretamente (só retorno e composição do Ibovespa, que
  poderia servir de proxy grosseiro — descartado para não importar viés de
  outro índice para dentro do sinal de momentum).

### Frequência e regra de rebalanceamento

Trimestral, com defasagem de execução de 2 pregões entre o cálculo
(fechamento da data de rebalance) e a data em que os novos pesos passam a
valer — simula o tempo operacional de implementar a carteira.

**Por que trimestral:** o sinal de momentum de 12 meses se move devagar —
rebalancear mensalmente eleva turnover (e custo) sem ganho relevante de
responsividade; rebalancear semestralmente/anualmente deixa a carteira
desatualizada por tempo demais dado que o efeito momentum tem meia-vida de
alguns meses. Trimestral é o padrão mais comum entre índices de momentum
institucionais (MSCI, S&P) por esse motivo.

### Tratamento de eventos corporativos

- Proventos, desdobramentos e grupamentos: **já incorporados** na série de
  retorno total ajustado fornecida — nenhum ajuste adicional necessário.
- Delisting / saída de um papel da base: a posição é liquidada ao último
  retorno disponível (sem penalidade adicional) e o caixa fica em CDI até
  o próximo rebalanceamento (ver `src/corporate_actions.py` para a
  justificativa completa e a alternativa descartada).
- **Limitação declarada**: sem mapeamento de código antigo → novo em
  incorporações, toda saída é tratada como delisting simples — pode
  subestimar levemente o retorno em casos de incorporação vantajosa
  (troca de ações). Fica listado como saída no log de delistings do
  backtest para auditoria.

### Controle e estimativa de turnover

Medido a cada rebalanceamento como `Σ|peso_novo − peso_antigo_pós_drift| / 2`.
Controlado por dois mecanismos: a banda de turnover na seleção (acima) e
uma tolerância mínima de ajuste (`min_weight_change`) que ignora variações
de peso irrelevantes. Turnover anualizado = turnover médio por
rebalanceamento × número de rebalanceamentos por ano.

## 2. Métricas de desempenho

Todas as métricas mínimas pedidas estão implementadas em `src/metrics.py`:
retorno acumulado e anualizado, volatilidade anualizada, Sharpe, máximo
drawdown, turnover anualizado, tracking error, beta vs. Ibovespa, active
share e atribuição de performance (`src/attribution.py`).

**Métricas próprias adicionadas**, e por quê:
- **Sortino e Calmar**: momentum tem risco de cauda assimétrico
  ("momentum crashes" em reversões bruscas de mercado, bem documentado na
  literatura) — volatilidade simétrica (Sharpe) não captura isso; Sortino
  pune só o downside, Calmar relaciona retorno a drawdown diretamente.
- **Hit rate mensal vs. Ibovespa**: métrica de discurso comercial — "com
  que frequência o produto bate o benchmark", mais intuitiva para o
  investidor final que Sharpe ou tracking error.
- **Up/down capture**: mostra SE o fator entrega o prêmio esperado e ONDE
  — idealmente captura mais alta que baixa; se for o contrário, é sinal de
  que o índice está mal calibrado ou que o período testado não favoreceu o
  fator.

## 3. Análise de sensibilidade e robustez

`src/sensitivity.py` oferece três checagens:

1. **Grade de parâmetros** (`parameter_grid_sensitivity`, chamada em
   `scripts/run_backtest.py`): reroda o backtest variando lookback e custo de
   transação, um de cada vez, comparando Sharpe/retorno. Se o resultado for
   muito sensível a uma escolha de calibração fina (ex.: 252 vs. 315 dias
   de lookback), é sinal de overfitting ao histórico específico.
2. **Estabilidade em janelas sucessivas** (`expanding_window_stability`,
   chamada em `scripts/run_backtest.py`): divide o backtest em 4
   subperíodos e compara Sharpe/retorno entre eles, para checar se o
   desempenho é consistente ao longo do tempo ou concentrado em uma janela
   específica.
3. **Busca em grade com validação treino/teste**
   (`train_test_grid_search`, script dedicado
   `scripts/run_train_test_split.py`): otimiza `lookback_days`,
   `entry_percentile` e `rebalance_freq` **ao mesmo tempo** na primeira
   metade da amostra (treino) e mede, **sem reotimizar**, o Sharpe da
   combinação vencedora na segunda metade (teste) — comparando contra o
   Sharpe do modelo DEFAULT (nunca ajustado) no mesmo teste. Isso responde
   a uma pergunta que a checagem (1) não responde sozinha: mesmo que nenhum
   parâmetro isolado pareça "frágil", a *combinação* escolhida pela grade
   pode ainda estar ajustada a ruído específico do treino — só um teste
   cego fora da amostra revela isso. Com os dados sintéticos deste
   ambiente, o modelo "otimizado" pela grade **não bate** o default fora da
   amostra (Sharpe de teste menor, apesar do Sharpe de treino mais alto) —
   evidência de que a grade capturou ruído do treino, não um padrão
   robusto, e reforça manter os parâmetros default (fixados por
   julgamento/literatura, não por busca) ao rodar contra os dados reais do
   case. Rode `python scripts/run_train_test_split.py` para reproduzir com
   os CSVs reais.

## 4. Capacidade do produto

`src/capacity.py` implementa a fórmula de capacidade
(`ADTV × participação_máxima × dias / peso`). Quando
`data/liquidez_mercado.csv` existe, `run_backtest.py` calcula a capacidade
em cada rebalance (`output/liquidez_carteira.csv`, junto com ADTV e market
cap da carteira) e por papel na carteira atual
(`output/capacidade_ultima_carteira.csv`), com o ADTV real do COTAHIST.

## O que ficou de fora (priorização declarada)

- Dados reais: não roda contra os CSVs do case porque eles não estavam
  disponíveis nesta sessão (ver aviso no topo).
- Fontes externas complementares (ex.: volume B3, classificação setorial
  para uma atribuição Brinson completa): não buscadas por falta de acesso
  a dados de mercado neste ambiente — não por decisão metodológica.
- Market cap com cobertura parcial: o mapeamento ticker → CNPJ do
  notebook só aceita matches de qualidade ALTO/MEDIO, units (final 11) não
  têm market cap, e o capital social da CVM vai até 12/2025. Papéis sem
  market cap não são filtrados por tamanho (ver `cobertura_liquidez.csv`).
- Dashboard interativo (Streamlit): optou-se por HTML estático
  autocontido, mais simples de entregar e abrir sem servidor rodando;
  trade-off é a ausência de filtros interativos por período.
- Apresentação de slides e leitura de produto (papel no portfólio, taxa de
  administração, tamanho de posição): são entregáveis separados do código,
  não cobertos por este pacote Python.

## Requisitos

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.4
```
