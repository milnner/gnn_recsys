#!/usr/bin/env bash
set -euo pipefail

# Padrões
RUNNER="./run_exp.sh"
OUTPUT_DIR="runs"
WANDB_MODE="disabled"
NUM_RUNS=3
BASE_SEED=42
DRY_RUN=false

# Parâmetros do experimento
DATASET=""
ROOT=""
MODEL=""
BATCH_SIZE=""
BASE_CONFIG=""

# ID -> "neg|perm|ratio|norm|fn"
ALL_IDS=(a1 a2 a3 a4 a5 b1 b2 b3 c1 c2 d1 d2)
declare -A EXP_SPEC=(
  [a1]="uniform||16|true|true"
  [a2]="mixed|0.25|16|true|true"
  [a3]="mixed|0.5|16|true|true"
  [a4]="mixed|0.75|16|true|true"
  [a5]="perm||16|true|true"
  [b1]="uniform||16|true|false"
  [b2]="mixed|0.5|16|true|false"
  [b3]="perm||16|true|false"
  [c1]="uniform||16|false|true"
  [c2]="mixed|0.5|16|false|true"
  [d1]="uniform||1|true|true"
  [d2]="mixed|0.5|1|true|true"
)

declare -A SELECTED=()
declare -A EXCLUDED=()

require_value() {
  [[ $# -ge 2 ]] || { echo "Erro: $1 requer um valor." >&2; usage; }
}

list_exps() {
  printf "%-4s | %-8s | %-6s | %-5s | %-5s | %-5s\n" "ID" "sampling" "perm_r" "ratio" "norm" "fn"
  printf "%s\n" "-----+----------+--------+-------+-------+------"
  for id in "${ALL_IDS[@]}"; do
    IFS='|' read -r neg perm ratio norm fn <<< "${EXP_SPEC[$id]}"
    printf "%-4s | %-8s | %-6s | %-5s | %-5s | %-5s\n" "$id" "$neg" "${perm:--}" "$ratio" "$norm" "$fn"
  done
}

usage() {
  cat <<EOF
Uso: $0 [obrigatórios] [seleção de experimentos] [opcionais]

Obrigatórios:
  --dataset STR         Nome do dataset (ex: ml32m, ml1m, Electronics)
  --root PATH           Diretório raiz dos dados (ex: ./data)
  --model STR           Nome do modelo (ex: hetero_sage, hetero_gatv2)
  --batch-size INT      Tamanho do batch (ex: 100000)
  --base-config PATH    Caminho do JSON base

Seleção atômica e presets (combináveis):
  --exp ID[,ID...]      Executa IDs específicos (ex: --exp b1,b2,c1)
  --exclude ID[,ID...]  Remove IDs da seleção atual (ex: --all --exclude a1,d1)
  --all                 Seleciona todos os 12 experimentos
  --block-a             Seleciona a1..a5 (RQ1: curva perm_ratio)
  --block-b             Seleciona b1..b3 (RQ2: filter_fn=false)
  --block-c             Seleciona c1..c2 (RQ3: normalize_output=false)
  --block-d             Seleciona d1..d2 (RQ1/3: ratio=1)
  --core                Seleciona a1,b1,a3,b2 (núcleo 4 configs para 2ª GNN)
  --list                Lista todos os IDs atômicos e sai

Opcionais:
  --runner PATH         Script base (padrão: ./run_exp.sh)
  --base-seed INT       Seed inicial (padrão: 42)
  --num-runs INT        Número de seeds por config (padrão: 3)
  --wandb-mode STR      online|offline|disabled (padrão: disabled)
  --output-dir PATH     Diretório de saída (padrão: runs)
  --dry-run             Imprime os comandos sem executar
EOF
  exit 1
}

add_ids() {
  local raw="$1"
  local target="$2" # SELECTED ou EXCLUDED
  IFS=',' read -ra ids <<< "$raw"
  for id in "${ids[@]}"; do
    if [[ -z "${EXP_SPEC[$id]+x}" ]]; then
      echo "Erro: ID de experimento desconhecido: '$id'. Use --list para ver os válidos." >&2
      exit 1
    fi
    if [[ "$target" == "SELECTED" ]]; then
      SELECTED["$id"]=1
    else
      EXCLUDED["$id"]=1
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)         list_exps; exit 0 ;;
    --runner)       RUNNER="$2"; shift 2 ;;
    --dataset)      DATASET="$2"; shift 2 ;;
    --root)         ROOT="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --base-config)  BASE_CONFIG="$2"; shift 2 ;;
    --base-seed)    BASE_SEED="$2"; shift 2 ;;
    --num-runs)     NUM_RUNS="$2"; shift 2 ;;
    --wandb-mode)   WANDB_MODE="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --exp)          require_value "$@"; add_ids "$2" SELECTED; shift 2 ;;
    --exclude)      add_ids "$2" EXCLUDED; shift 2 ;;
    --all)          add_ids "$(IFS=,; printf '%s' "${ALL_IDS[*]}")" SELECTED; shift ;;
    --block-a)      add_ids "a1,a2,a3,a4,a5" SELECTED; shift ;;
    --block-b)      add_ids "b1,b2,b3" SELECTED; shift ;;
    --block-c)      add_ids "c1,c2" SELECTED; shift ;;
    --block-d)      add_ids "d1,d2" SELECTED; shift ;;
    --core)         add_ids "a1,b1,a3,b2" SELECTED; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage ;;
    *) echo "Argumento desconhecido: $1" >&2; usage ;;
  esac
done

for var in DATASET ROOT MODEL BATCH_SIZE BASE_CONFIG; do
  if [[ -z "${!var}" ]]; then
    echo "Erro: parâmetro obrigatório ausente ($var)." >&2
    usage
  fi
done

# Monta fila final na ordem canônica, aplicando exclusões
QUEUE=()
for id in "${ALL_IDS[@]}"; do
  if [[ -n "${SELECTED[$id]+x}" && -z "${EXCLUDED[$id]+x}" ]]; then
    QUEUE+=("$id")
  fi
done

if [[ ${#QUEUE[@]} -eq 0 ]]; then
  echo "Erro: nenhum experimento selecionado. Use --exp, --block-*, --core ou --all." >&2
  usage
fi

echo "Fila de execução (${#QUEUE[@]} configs): ${QUEUE[*]}"

for id in "${QUEUE[@]}"; do
  IFS='|' read -r neg perm ratio norm fn <<< "${EXP_SPEC[$id]}"

  cmd=(
    "$RUNNER"
    --dataset "$DATASET"
    --root "$ROOT"
    --model "$MODEL"
    --batch-size "$BATCH_SIZE"
    --base-config "$BASE_CONFIG"
    --base-seed "$BASE_SEED"
    --num-runs "$NUM_RUNS"
    --wandb-mode "$WANDB_MODE"
    --output-dir "$OUTPUT_DIR"
    --neg-sampling "$neg"
    --ratio "$ratio"
    --normalize-output "$norm"
    --filter-fn "$fn"
  )

  [[ -n "$perm" ]] && cmd+=(--perm-ratio "$perm")

  if [[ "$DRY_RUN" == true ]]; then
    printf '[DRY-RUN][%s] ' "$id"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    echo ">>> [$id] neg=${neg} perm=${perm:--} ratio=${ratio} norm=${norm} fn=${fn}"
    "${cmd[@]}"
  fi
done