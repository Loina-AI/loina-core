#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target=chat
case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: ./download_model.sh [chat|vision|speech|image|all|PACKAGE_ID] [options]

Downloads verified catalog models into gguf/ beside the Loina binaries.
Default target: chat. LOINA_GGUF_DIR overrides the model folder.
Conversations and KV live separately in ~/.loina/kvcache.

Options:
  --model-dir PATH     Choose another folder for model files.
  --cache-dir PATH     Choose another session and KV folder.
  --import DIRECTORY   Install existing model files instead of downloading.
  --list               List packages and installation state.
EOF
        exit 0 ;;
    --list) shift; exec "$root/loina" --models "$@" ;;
    ""|--*) ;;
    *) target=$1; shift ;;
esac
if [ ! -x "$root/loina" ]; then
    echo "Build Loina first: make -C \"$root\"" >&2
    exit 1
fi
exec "$root/loina" --download "$target" "$@"
