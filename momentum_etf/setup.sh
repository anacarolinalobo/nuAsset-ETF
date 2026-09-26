#!/usr/bin/env bash
# Prepara o ambiente local para rodar o backtest com dados reais.
#
# Uso (rodar de dentro de momentum_etf/, na sua máquina local):
#   ./setup.sh
#
# O que faz:
#   1. Cria/ativa um virtualenv (.venv)
#   2. Instala requirements.txt
#   3. Copia ibov_composicao.csv e benchmarks_diarios.csv da raiz do repo
#      para data/ (eles já vêm versionados no repositório)
#   4. Avisa se acoes_retornos.csv ainda não estiver em data/ — esse
#      arquivo não é versionado (tamanho) e precisa ser colocado manualmente
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

echo "== 1/4: virtualenv =="
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

echo "== 2/4: dependências =="
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo "== 3/4: copiando CSVs já disponíveis no repo =="
mkdir -p "$DATA_DIR"
for f in ibov_composicao.csv benchmarks_diarios.csv; do
    if [ -f "$REPO_ROOT/$f" ]; then
        cp -f "$REPO_ROOT/$f" "$DATA_DIR/$f"
        echo "  copiado: $f"
    else
        echo "  AVISO: $REPO_ROOT/$f não encontrado (esperado na raiz do repo)"
    fi
done

echo "== 4/4: checando acoes_retornos.csv =="
if [ -f "$DATA_DIR/acoes_retornos.csv" ]; then
    echo "  OK: data/acoes_retornos.csv presente."
    echo
    echo "Tudo pronto. Rode:"
    echo "  source .venv/bin/activate"
    echo "  python scripts/run_backtest.py"
else
    echo "  FALTA: data/acoes_retornos.csv"
    echo
    echo "Esse arquivo (colunas: date, ticker, retorno) não está versionado no"
    echo "repositório e precisa ser colocado manualmente em:"
    echo "  $DATA_DIR/acoes_retornos.csv"
    echo
    echo "Depois de colocá-lo, rode:"
    echo "  source .venv/bin/activate"
    echo "  python scripts/run_backtest.py"
    exit 1
fi
