#!/usr/bin/env bash
#
# DuxxDB — one-shot installer (Linux + macOS).
#
# Detects OS + arch, downloads the matching binary archive from the
# latest GitHub Release, and installs the binaries into /usr/local/bin
# (or $DUXX_INSTALL_DIR if you set it).
#
# Quickstart:
#   curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh | sh
#
# Pin to a specific version:
#   curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh | DUXX_VERSION=v0.1.0 sh
#
# Install to a non-default prefix (e.g. ~/bin without sudo):
#   curl -fsSL https://raw.githubusercontent.com/bankyresearch/duxxdb/master/packaging/scripts/install.sh | DUXX_INSTALL_DIR="$HOME/bin" sh
#
# What this script does NOT do:
#   * Install or enable a systemd unit (use the .deb / .rpm for that).
#   * Create a duxxdb system user.
#   * Open any firewall ports.
# It just drops the binaries on $PATH so you can `duxx-server --help`.

set -euo pipefail

REPO="bankyresearch/duxxdb"
VERSION="${DUXX_VERSION:-latest}"
INSTALL_DIR="${DUXX_INSTALL_DIR:-/usr/local/bin}"

err() { printf 'install.sh: \033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf 'install.sh: %s\n' "$*"; }

# ---- detect platform --------------------------------------------------------
uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "$uname_s" in
  Linux)  os="linux"  ;;
  Darwin) os="macos"  ;;
  *) err "unsupported OS: $uname_s. On Windows, download the zip from GitHub Releases." ;;
esac

case "$uname_m" in
  x86_64|amd64)  arch="x86_64"  ;;
  arm64|aarch64) arch="aarch64" ;;
  *) err "unsupported arch: $uname_m" ;;
esac

target="${arch}-${os}"
log "detected platform: $target"

# ---- pick a download tool ---------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
  fetch_stdout() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q -O "$2" "$1"; }
  fetch_stdout() { wget -q -O - "$1"; }
else
  err "neither curl nor wget is installed. Install one and retry."
fi

# ---- resolve version --------------------------------------------------------
if [ "$VERSION" = "latest" ]; then
  VERSION="$(fetch_stdout "https://api.github.com/repos/${REPO}/releases/latest" \
             | grep -m1 '"tag_name"' \
             | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
  if [ -z "$VERSION" ]; then
    err "could not resolve latest release. Pin a version: DUXX_VERSION=v0.1.0"
  fi
  log "resolved latest release: $VERSION"
fi

# ---- download archive -------------------------------------------------------
archive="duxxdb-${VERSION}-${target}.tar.gz"
url="https://github.com/${REPO}/releases/download/${VERSION}/${archive}"

tmp="$(mktemp -d -t duxxdb-install.XXXXXX)"
trap 'rm -rf "$tmp"' EXIT

log "downloading $url"
fetch "$url" "$tmp/$archive" \
  || err "download failed. Check that release $VERSION ships an archive for $target."

# ---- verify checksum if available -------------------------------------------
sums="duxxdb-${VERSION}-SHA256SUMS"
if fetch "https://github.com/${REPO}/releases/download/${VERSION}/${sums}" "$tmp/$sums" 2>/dev/null; then
  log "verifying SHA-256"
  ( cd "$tmp" && grep " ${archive}\$" "$sums" | sha256sum -c - ) \
    || err "checksum mismatch. Aborting."
else
  log "no SHA256SUMS file in release; skipping checksum verification"
fi

# ---- extract + install ------------------------------------------------------
log "extracting"
tar -xzf "$tmp/$archive" -C "$tmp"

# Archive layout: duxxdb-<version>-<target>/{duxx-server, duxx-grpc, duxx-mcp, duxx-export, ...}
src="$tmp/duxxdb-${VERSION}-${target}"
[ -d "$src" ] || err "unexpected archive layout (no $src)"

log "installing binaries to $INSTALL_DIR"
need_sudo=""
if [ ! -w "$INSTALL_DIR" ]; then
  if command -v sudo >/dev/null 2>&1; then
    need_sudo="sudo"
  else
    err "cannot write to $INSTALL_DIR and sudo is unavailable. Set DUXX_INSTALL_DIR=\$HOME/bin and retry."
  fi
fi

mkdir_priv()  { $need_sudo mkdir -p "$1"; }
install_priv() { $need_sudo install -m 0755 "$1" "$2"; }

mkdir_priv "$INSTALL_DIR"
for bin in duxx-server duxx-grpc duxx-mcp duxx-export; do
  if [ -f "$src/$bin" ]; then
    install_priv "$src/$bin" "$INSTALL_DIR/$bin"
    log "  installed $bin"
  fi
done

# ---- post-install hint ------------------------------------------------------
cat <<EOF

$(duxx-server --version 2>/dev/null || printf 'duxx-server')

Installed to: $INSTALL_DIR
Verify:       duxx-server --help

Quickstart:
  # Pick ONE of:

  # 1) Localhost-only, no auth, in-memory:
  duxx-server --addr 127.0.0.1:6379

  # 2) Localhost-only, persistent dir-storage:
  mkdir -p \$HOME/.duxxdb
  duxx-server --addr 127.0.0.1:6379 --storage dir:\$HOME/.duxxdb

  # 3) Production-style with auth + metrics:
  export DUXX_TOKEN=\$(openssl rand -hex 32)
  echo "your token: \$DUXX_TOKEN"   # save this somewhere safe
  duxx-server \\
    --addr 0.0.0.0:6379 \\
    --storage dir:\$HOME/.duxxdb \\
    --metrics-addr 127.0.0.1:9100

Connect from another shell:
  redis-cli -p 6379 PING        # +PONG (auth-free; PING is allowed pre-AUTH)

For systemd / .deb / .rpm install:
  https://github.com/${REPO}/blob/master/docs/INSTALLATION.md

EOF
