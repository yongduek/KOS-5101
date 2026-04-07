#!/usr/bin/env bash
# =============================================================================
#  exe_notebooks.sh — IRT 노트북 일괄 실행 스크립트 (macOS / Linux)
#
#  사용법:
#    bash exe_notebooks.sh                  # 모든 노트북 실행
#    bash exe_notebooks.sh IRT_P2_PCM.ipynb # 특정 노트북만 실행
#    bash exe_notebooks.sh --timeout 180    # 셀 타임아웃(초) 지정
# =============================================================================

set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMEOUT=120
SPECIFIC_NB=""

# ── 인수 파싱 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        *.ipynb)    SPECIFIC_NB="$1"; shift ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  IRT Notebook Executor"
echo "  Directory : $SCRIPT_DIR"
echo "  Timeout   : ${TIMEOUT}s per cell"
echo "============================================================"
echo ""

# ── 방법 1: jupyter nbconvert 가 있으면 사용 ─────────────────────────────
if command -v jupyter &>/dev/null; then
    echo "▶  jupyter nbconvert 발견 → 사용합니다."
    echo ""

    run_one_jupyter() {
        local nb="$1"
        echo -n "  실행 중: $nb ... "
        jupyter nbconvert \
            --to notebook \
            --execute \
            --allow-errors \
            --inplace \
            --ExecutePreprocessor.timeout="$TIMEOUT" \
            "$nb" 2>/dev/null \
            && echo "✅  완료" \
            || echo "⚠️  일부 오류 (출력 저장됨)"
    }

    if [[ -n "$SPECIFIC_NB" ]]; then
        run_one_jupyter "$SPECIFIC_NB"
    else
        for nb in *.ipynb; do
            [[ -f "$nb" ]] && run_one_jupyter "$nb"
        done
    fi

# ── 방법 2: jupyter 없으면 run_notebooks.py 사용 ─────────────────────────
elif command -v python3 &>/dev/null && [[ -f "run_notebooks.py" ]]; then
    echo "▶  jupyter 없음 → run_notebooks.py (내장 실행기) 사용."
    echo ""

    if [[ -n "$SPECIFIC_NB" ]]; then
        python3 run_notebooks.py --timeout "$TIMEOUT" "$SPECIFIC_NB"
    else
        python3 run_notebooks.py --timeout "$TIMEOUT"
    fi

elif command -v python &>/dev/null && [[ -f "run_notebooks.py" ]]; then
    if [[ -n "$SPECIFIC_NB" ]]; then
        python run_notebooks.py --timeout "$TIMEOUT" "$SPECIFIC_NB"
    else
        python run_notebooks.py --timeout "$TIMEOUT"
    fi

# ── 방법 3: 아무것도 없으면 설치 안내 ───────────────────────────────────
else
    echo "❌  오류: jupyter 또는 python3을 찾을 수 없습니다."
    echo ""
    echo "  다음 중 하나를 설치하세요:"
    echo ""
    echo "  # Anaconda / conda 환경 (권장)"
    echo "  conda install -c conda-forge jupyter cmdstanpy"
    echo ""
    echo "  # pip 환경"
    echo "  pip install jupyter nbconvert cmdstanpy"
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo "  완료! 노트북 파일(.ipynb)을 열어 결과를 확인하세요."
echo "============================================================"
