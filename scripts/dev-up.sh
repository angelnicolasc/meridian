#!/usr/bin/env bash
# Bring the dev environment up. Idempotent.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f /.dockerenv || -n "${DEVCONTAINER:-}" ]]; then
    echo "==> Inside devcontainer — running post-create."
    bash .devcontainer/post-create.sh
    exit 0
fi

if ! command -v devcontainer >/dev/null 2>&1; then
    cat <<'EOM' >&2
ERROR: `devcontainer` CLI not found.

Install:    npm install -g @devcontainers/cli
Or:         open this folder in VS Code with the Dev Containers extension.
EOM
    exit 1
fi

echo "==> Bringing up devcontainer (this may take several minutes the first time)"
devcontainer up --workspace-folder "$(pwd)"

echo "==> Running post-create hook inside container"
devcontainer exec --workspace-folder "$(pwd)" -- bash .devcontainer/post-create.sh
