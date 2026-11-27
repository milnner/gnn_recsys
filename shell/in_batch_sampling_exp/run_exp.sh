#!/usr/bin/env bash
set -euo pipefail

# Valores padrão
OUTPUT_DIR="runs"
WANDB_MODE="disabled"
NUM_RUNS=3
SCORE_SCALE=""

# Parâmetros obrigatórios do experimento
DATASET=""
ROOT=""
RATIO=""
NEG_SAMPLING=""
FILTER_FN=""
MODEL=""
NORM_OUT=""
BATCH_SIZE=""
BASE_CONFIG=""
PERM_RATIO=""
BASE_SEED=""

usage() {
  cat <<EOF
Uso: $0 [opções]

Obrigatórios:
  --dataset STR           Nome do dataset (ex: All_Beauty)
  --root PATH             Diretório raiz dos dados (ex: ./data)
  --ratio INT             Razão de negativos por positivo (ex: 4)
  --neg-sampling STR      Estratégia de amostragem negativa (ex: uniform, mixed)
  --filter-fn BOOL        Filtrar falsos negativos (true|false)
  --model STR             Nome do modelo (ex: sage, lightgcn)
  --normalize-output BOOL Normalizar saída do modelo (true|false)
  --batch-size INT        Tamanho do batch (ex: 2048)
  --base-seed INT         Seed inicial, incrementada a cada run
  --base-config PATH      JSON base

Opcionais:
  --wandb-mode STR        online|offline|disabled (padrão: disabled)
  --num-runs INT          Quantidade de runs (padrão: 3)
  --score-scale FLOAT     Sobrescreve train.score_scale (padrão: 10.0 se normalize-output=true, senão 1.0)
  --output-dir PATH       Diretório de saída do main.py (padrão: runs)
  --perm-ratio FLOAT      Razão de arestas negativas advindas de permutação

EOF
  exit 1
}

# Parse dos argumentos
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config)      BASE_CONFIG="$2"; shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
    --wandb-mode)       WANDB_MODE="$2"; shift 2 ;;
    --num-runs)         NUM_RUNS="$2"; shift 2 ;;
    --base-seed)        BASE_SEED="$2"; shift 2 ;;
    --score-scale)      SCORE_SCALE="$2"; shift 2 ;;
    --dataset)          DATASET="$2"; shift 2 ;;
    --root)             ROOT="$2"; shift 2 ;;
    --ratio)            RATIO="$2"; shift 2 ;;
    --perm-ratio)       PERM_RATIO="$2"; shift 2 ;;
    --neg-sampling)     NEG_SAMPLING="$2"; shift 2 ;;
    --filter-fn)        FILTER_FN="$2"; shift 2 ;;
    --model)            MODEL="$2"; shift 2 ;;
    --normalize-output) NORM_OUT="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)          usage ;;
    *) echo "Argumento desconhecido: $1" >&2; usage ;;
  esac
done

# Validação dos obrigatórios
for var in DATASET ROOT RATIO NEG_SAMPLING FILTER_FN MODEL NORM_OUT BATCH_SIZE BASE_CONFIG BASE_SEED; do
  if [[ -z "${!var}" ]]; then
    echo "Erro: parâmetro obrigatório ausente ($var)." >&2
    usage
  fi
done

for var in RATIO BATCH_SIZE NUM_RUNS BASE_SEED; do
  if ! [[ "${!var}" =~ ^-?[0-9]+$ ]]; then
    echo "Erro: ($var) deve ser inteiro." >&2; usage
  fi
done

if [[ "$NEG_SAMPLING" == "mixed" && -z "$PERM_RATIO" ]]; then
  echo "Erro: --perm-ratio é obrigatório quando --neg-sampling=mixed." >&2
  usage
fi



# Define score_scale automaticamente se não foi informado
if [[ -z "$SCORE_SCALE" ]]; then
  if [[ "$NORM_OUT" == "true" ]]; then
    SCORE_SCALE="10.0"
  else
    SCORE_SCALE="1.0"
  fi
fi

# Pasta temporária com limpeza automática ao sair (sucesso ou erro)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Iniciando ${NUM_RUNS} runs (configs temporárias em ${TMP_DIR})"

for ((i = 0; i < NUM_RUNS; i++)); do
  SEED=$((BASE_SEED + i))
  TMP_CFG="${TMP_DIR}/config_run_${i}_seed_${SEED}.json"

  jq \
    --arg ds "$DATASET" \
    --arg root "$ROOT" \
    --argjson ratio "$RATIO" \
    --arg perm "$PERM_RATIO" \
    --arg neg "$NEG_SAMPLING" \
    --argjson filter_fn "$FILTER_FN" \
    --arg model "$MODEL" \
    --argjson norm_out "$NORM_OUT" \
    --argjson bs "$BATCH_SIZE" \
    --argjson scale "$SCORE_SCALE" \
    --argjson seed "$SEED" \
    '.data.dataset = $ds
     | .data.root = $root
     | .sampler.ratio = $ratio
     | (if $perm == "" then . else .sampler.perm_ratio = ($perm | tonumber) end)
     | .sampler.negative_sampling = $neg
     | .sampler.filter_false_negatives = $filter_fn
     | .model.name = $model
     | .model.normalize_output = $norm_out
     | .train.batch_size = $bs
     | .train.score_scale = $scale
     | .train.seed = $seed' \
    "$BASE_CONFIG" > "$TMP_CFG"

  echo "========================================================"
  echo "Run $((i + 1))/${NUM_RUNS} | seed=${SEED} | dataset=${DATASET} | model=${MODEL}"
  echo "========================================================"

  pixi run python code/scripts/inbatch_negative_sampling_exp.py \
    --config "$TMP_CFG" \
    --output-dir "$OUTPUT_DIR" \
    --wandb-mode "$WANDB_MODE"
done

echo "Todas as ${NUM_RUNS} runs foram concluídas."