#!/usr/bin/env bash
# Ensure /Volumes/FileStore is mounted before the agent reads astro data.
# Used by Cursor hooks (sessionStart, beforeShellExecution, preToolUse).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/filestore.conf"

input=""
if [[ ! -t 0 ]]; then
  input="$(perl -MPOSIX -e 'alarm 1; local $/; print <> ' 2>/dev/null || true)"
fi

mounted() {
  [[ -d "${FILESTORE_VERIFY_PATH}" ]]
}

run_mount_command() {
  if [[ -z "${FILESTORE_SMB_URL:-}" ]]; then
    return 1
  fi

  # Cap AppleScript mount time so hooks do not hang on auth prompts.
  perl -MPOSIX -e '
    alarm shift @ARGV;
    exec @ARGV;
  ' 15 osascript \
    -e 'try' \
    -e "mount volume \"${FILESTORE_SMB_URL}\"" \
    -e 'end try' 2>/dev/null || true
}

mount_filestore() {
  if mounted; then
    return 0
  fi

  mkdir -p "${FILESTORE_MOUNT_POINT}" 2>/dev/null || true
  run_mount_command

  local i=0
  while (( i < FILESTORE_MOUNT_TIMEOUT )); do
    if mounted; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done

  return 1
}

should_attempt_mount() {
  if mounted; then
    return 1
  fi
  if [[ -z "${input}" ]]; then
    return 0
  fi
  if echo "${input}" | grep -qE 'FileStore|/Volumes/FileStore|ASTRO/AARO/NINA'; then
    return 0
  fi
  return 1
}

emit_allow() {
  if mounted; then
    echo '{"permission":"allow"}'
    return
  fi
  python3 - <<'PY'
import json
print(json.dumps({
    "permission": "allow",
    "agent_message": (
        "FileStore auto-mount failed. The share may need to be connected once in "
        "Finder (Go → Connect to Server → smb://Mac Studio/FileStore) so macOS "
        "can reuse saved credentials."
    ),
}))
PY
}

if should_attempt_mount; then
  mount_filestore || true
fi

if echo "${input}" | grep -qE '"command"|"tool_name"'; then
  emit_allow
fi

exit 0
