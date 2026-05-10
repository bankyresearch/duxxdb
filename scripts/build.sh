#!/usr/bin/env bash
# Windows + Git Bash build wrapper for DuxxDB (GNU toolchain path).
#
# Sets up PATH so cargo can find:
#   1. cargo / rustc itself (~/.cargo/bin)
#   2. WinLibs MinGW (gcc, ld, as, ar, dlltool, ...) installed by:
#        winget install BrechtSanders.WinLibs.POSIX.MSVCRT
#
# Usage:
#   scripts/build.sh check
#   scripts/build.sh build --release
#   scripts/build.sh test --workspace
#   scripts/build.sh run -p duxx-cli --example chatbot_memory

set -euo pipefail

CARGO_BIN="$HOME/.cargo/bin"
MINGW_BIN="$LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
PROTOC_BIN="$LOCALAPPDATA/Microsoft/WinGet/Packages/Google.Protobuf_Microsoft.Winget.Source_8wekyb3d8bbwe/bin"

# In Git Bash, $LOCALAPPDATA is a Windows-style path. Normalize via cygpath.
if command -v cygpath >/dev/null 2>&1; then
  MINGW_BIN="$(cygpath -u "$MINGW_BIN")"
  PROTOC_BIN="$(cygpath -u "$PROTOC_BIN")"
fi

if [[ ! -x "$MINGW_BIN/gcc.exe" ]]; then
  echo "ERROR: WinLibs MinGW not found at:" >&2
  echo "  $MINGW_BIN" >&2
  echo "Install with:" >&2
  echo "  winget install BrechtSanders.WinLibs.POSIX.MSVCRT --silent \\" >&2
  echo "    --accept-package-agreements --accept-source-agreements" >&2
  exit 1
fi

if [[ ! -x "$CARGO_BIN/cargo.exe" ]]; then
  echo "ERROR: cargo not found at $CARGO_BIN — install rustup first." >&2
  exit 1
fi

# protoc is optional (only needed for the duxx-grpc crate). Add to PATH
# if installed; cargo will surface a clear error if needed but absent.
if [[ -x "$PROTOC_BIN/protoc.exe" ]]; then
  export PATH="$PROTOC_BIN:$PATH"
fi

export PATH="$MINGW_BIN:$CARGO_BIN:$PATH"
exec cargo "$@"
