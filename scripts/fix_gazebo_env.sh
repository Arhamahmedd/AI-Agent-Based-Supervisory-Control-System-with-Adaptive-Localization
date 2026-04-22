#!/usr/bin/env bash
set -euo pipefail

# Remove snap core runtime library paths to avoid libc/libpthread conflicts.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  CLEANED_LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/snap/core' | paste -sd: -)"
  export LD_LIBRARY_PATH="$CLEANED_LD_LIBRARY_PATH"
fi

exec "$@"
