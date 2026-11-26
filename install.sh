#!/usr/bin/env bash
# -----------------------------------------------
# Open Source AI Platform — alias 등록
# -----------------------------------------------
# 한 번 실행하면 어디서든 platform-start 로 서버를 켠다.
#
#   ./install.sh && source ~/.bashrc
#   # zsh 이면: source ~/.zshrc
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${OAP_LOG:-/tmp/opensource_ai_platform.log}"

MARKER_BEGIN="# >>> opensource-ai-platform aliases >>>"
MARKER_END="# <<< opensource-ai-platform aliases <<<"

BLOCK=$(cat <<EOF
${MARKER_BEGIN}
# Open Source AI Platform (4-tab UI + shared camera hub)
alias platform-start='bash ${ROOT}/start.sh'
alias platform-start-bg='nohup bash ${ROOT}/start.sh > ${LOG} 2>&1 & echo started pid=\$!; sleep 1; tail -n 20 ${LOG}'
alias platform-stop='bash ${ROOT}/stop.sh'
alias platform-logs='tail -f ${LOG}'
alias platform-status='bash ${ROOT}/status.sh'
${MARKER_END}
EOF
)

upsert() {
  local rc="$1"
  if [[ ! -f "$rc" ]]; then
    touch "$rc"
  fi

  if grep -qF "${MARKER_BEGIN}" "$rc" 2>/dev/null; then
    echo "[INFO] ${rc} 의 alias 를 업데이트합니다."
    if sed --version >/dev/null 2>&1; then
      sed -i "\|${MARKER_BEGIN}|,\|${MARKER_END}|d" "$rc"
    else
      sed -i.bak "\|${MARKER_BEGIN}|,\|${MARKER_END}|d" "$rc"
      rm -f "${rc}.bak"
    fi
  fi

  printf '\n%s\n' "$BLOCK" >> "$rc"
  echo "[OK] ${rc}"
}

upsert "$HOME/.bashrc"
if [[ -f "$HOME/.zshrc" ]] || [[ "${SHELL:-}" == *zsh ]]; then
  upsert "$HOME/.zshrc"
fi

chmod +x \
  "$ROOT/install.sh" \
  "$ROOT/start.sh" \
  "$ROOT/stop.sh" \
  "$ROOT/status.sh"

echo ""
echo "[DONE] alias 등록 완료. 현재 셸에 적용:"
echo ""
echo "  source ~/.bashrc     # bash"
echo "  source ~/.zshrc      # zsh"
echo ""
echo "등록된 명령어:"
echo "  platform-start      시작 (foreground, http://<host>:8095)"
echo "  platform-start-bg   백그라운드 시작 (${LOG})"
echo "  platform-stop       중지"
echo "  platform-logs       백그라운드 로그"
echo "  platform-status     실행 상태"
echo ""
echo "설정: ${ROOT}/config.yaml (OAP_CONFIG 로 덮어쓰기 가능)"
