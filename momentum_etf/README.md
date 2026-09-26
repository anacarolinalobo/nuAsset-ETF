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
- **Fontes externas usadas**: dados de liquidez (volume financeiro
  negociado) e tamanho (valor de mercado) via arquivos públicos da B3
  (COTAHIST) e da CVM (Formulário de Referência — capital social), lidos
  pelos módulos `src/cotahist.py`, `src/fre_capital_social.py` e
  `src/market_cap.py`. Também não estavam anexados a esta sessão — o
  código está pronto contra o layout real desses arquivos (validado
  manualmente no `caseNuAsset.ipynb` original), mas não rodou contra dado
  de verdade aqui. Ver seção "Filtro de liquidez e tamanho" abaixo.

## Como rodar

```bash
pip install -r requirements.txt

# 1) smoke test com dados sintéticos (opcional, só para validar o pipeline
#    de ponta a ponta, incluindo o filtro de liquidez/market cap)
python scripts/make_sample_data.py
python scripts/build_market_data.py   # gera ADTV e market cap sintéticos

# 2) com os dados reais em data/ (substituindo os sintéticos, se gerados):
#    - acoes_retornos.csv, ibov_composicao.csv, benchmarks_diarios.csv
#    - data/cotahist/COTAHIST_A{ano}.TXT (2008-2026)
#    - data/fre_cia_aberta_{ano}/fre_cia_aberta_capital_social_{ano}.csv (2010-2026)
python scripts/build_market_data.py   # opcional; sem isso, cai no proxy antigo
python scripts/run_backtest.py

# 3) validação treino/teste da grade de parâmetros (checagem de overfitting,
#    pode levar vários minutos — 1 backtest por combinação da grade)
python scripts/run_train_test_split.py

# 4) testes unitários
pytest -q
```

`build_market_data.py` é opcional: se `data/derived/adtv.csv` e
`market_cap.csv` não existirem, `run_backtest.py` roda igual, só que com
o filtro de liquidez de fallback (presença de retorno, sem dado externo).

Saídas em `output/`: `dashboard.html` (relatório visual), `summary_metrics.csv`,
`daily_returns.csv`, `turnover_history.csv`, `attribution.csv`,
`sensitivity_lookback.csv`, `sensitivity_cost.csv`. Saídas de
`scripts/run_train_test_split.py` em `output/train_test_split/`:
`grid_treino_teste.csv`, `resumo.csv`. Em `data/derived/`:
`ticker_cnpj_mapping.csv` (revisar antes de confiar — ver seção de
limitações), `adtv.csv`, `market_cap.csv`.

## Estrutura do código

```
src/
  config.py            parâmetros da metodologia, centralizados
  data_loader.py        carga e limpeza dos 3 CSVs; checagem de dado suspeito
  cotahist.py             parser dos arquivos de pregão B3 (preço + volume)
  fre_capital_social.py   ações em circulação (Formulário de Referência CVM)
  company_mapping.py      matching ticker <-> CNPJ via nome (fuzzy)
  market_cap.py            combina preço x ações em circulação = valor de mercado
  universe.py            universo elegível: liquidez/tamanho ANTES do momentum
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
  build_market_data.py     parseia COTAHIST + FRE, gera ADTV/market cap/mapeamento
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

A elegibilidade roda em **dois estágios**, sempre usando apenas dados
disponíveis até a própria data de cálculo:

**Estágio 1 — filtro de liquidez e tamanho** (`src/universe.py::_liquidity_size_filter`),
aplicado **antes de qualquer coisa relacionada a momentum** — ser
"negociável em tamanho razoável" precisa ser uma pergunta independente de
"teve bom desempenho recente", senão o índice fica enviesado a comprar
justamente os papéis ilíquidos que mais dispararam por pouca profundidade
de book, o caso clássico de resultado de backtest bonito e impossível de
implementar:

1. **ADTV** (volume financeiro médio negociado, janela móvel de
   `liquidity_lookback_days` = 63 pregões, calculada a partir do COTAHIST
   em `src/cotahist.py::compute_adtv`) — exclui o terço menos líquido da
   seção transversal do dia (`adtv_min_percentile` = 0.30).
2. **Valor de mercado** (preço de fechamento real × ações em circulação,
   `src/market_cap.py`) — exclui o quinto menor em tamanho
   (`market_cap_min_percentile` = 0.20).

Os cortes são por **percentil da seção transversal do dia**, não valor
absoluto em R$: um piso nominal fixo perderia sentido ao longo de
2010-2026 (inflação, crescimento do mercado) e teria que ser recalibrado
a cada ano; o corte relativo se mantém comparável no tempo. Um piso
absoluto opcional (`min_adtv_reais`, `min_market_cap_reais`) pode ser
somado por cima quando fizer sentido (ex.: garantir um mínimo de R$
negociado por dia que qualquer AUM-alvo do fundo precisaria conseguir
executar), mas fica desligado por padrão.

Quando ADTV/market cap não estão disponíveis (dado externo ausente, ou
ticker sem par confiável na base de CNPJ — ver "Limitações do
mapeamento" abaixo), o estágio 1 não penaliza o papel: ele segue para o
estágio 2 avaliado só pelo proxy interno, para que ausência de dado
externo nunca vire exclusão silenciosa.

**Estágio 2 — histórico e proxy de atividade** (`_history_filter`, a
lógica já existente antes desta mudança), roda sobre o que sobrou do
estágio 1:

1. Tem pelo menos `lookback_days + skip_days` (padrão: 273) pregões de
   histórico — para que o sinal de 12 meses esteja plenamente formado.
2. Ainda está "viva": teve ao menos um retorno observado nos últimos
   `liquidity_window_days` (126) pregões — se não, presume-se deslistada.
3. Proxy de liquidez interno (fração de pregões com retorno observado
   >= `min_active_ratio`, 90%) — continua ativo mesmo com ADTV real
   disponível, como uma segunda rede de segurança contra dado de preço
   sem negócio de fato por trás (ver `data_loader.flag_suspicious_returns`).

A ação sai do universo automaticamente quando seu histórico de retorno
acaba, quando deixa de passar no filtro de liquidez/tamanho, ou quando
cai abaixo do proxy de atividade.

**Por que isso evita survivorship bias:** como `acoes_retornos.csv` inclui
ações que saíram de negociação até o último dia em que negociaram, o
universo em cada data histórica é calculado exatamente como teria sido
calculado *naquele momento* — nada é removido retroativamente por "ter
morrido depois".

**Limitações do mapeamento ticker↔CNPJ** (`src/company_mapping.py`): não
existe chave exata em comum entre COTAHIST (ticker + nome truncado em 12
caracteres) e o FRE/CVM (CNPJ + nome oficial completo) — o matching é por
similaridade de nome normalizado (fuzzy), com um limiar mínimo de
confiança. Nomes truncados colidem com frequência entre empresas do
mesmo grupo econômico ou entre diferentes classes de ação da mesma
controladora. `scripts/build_market_data.py` exporta
`data/derived/ticker_cnpj_mapping.csv` com a pontuação de similaridade de
cada match para revisão manual — este entregável NÃO valida esse
mapeamento linha a linha contra uma fonte de verdade (seria o próximo
passo antes de usar o filtro em produção).

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
(`ADTV × participação_máxima × dias / peso`), mas **não é calculada** no
`run_backtest.py` porque a base fornecida não traz volume financeiro
negociado — só retorno. Buscar essa série (ex.: COTAHIST da B3) é o
próximo passo natural antes de qualquer decisão de tamanho de produto; a
função está pronta para receber um `pd.Series` de ADTV por ticker assim
que disponível, sem mudar a interface.

## O que ficou de fora (priorização declarada)

- Dados reais: não roda contra os CSVs do case nem contra COTAHIST/FRE de
  verdade porque nenhum deles estava disponível nesta sessão (ver aviso
  no topo) — todo o código de liquidez/market cap foi validado só com
  dados sintéticos gerados por `scripts/make_sample_data.py`.
- Validação manual do mapeamento ticker↔CNPJ (`ticker_cnpj_mapping.csv`):
  o matching é fuzzy e fica exportado para revisão, mas essa revisão
  linha a linha não foi feita aqui — ver "Limitações do mapeamento" acima.
- Classificação setorial para uma atribuição Brinson completa (a
  atribuição implementada é por contribuição de ativo, não por setor) —
  não buscada por falta de acesso a dados de mercado neste ambiente.
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
