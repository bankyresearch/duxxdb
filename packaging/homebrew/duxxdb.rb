# Homebrew formula for DuxxDB.
#
# This file is the source-of-truth that gets pushed to the
# `bankyresearch/homebrew-duxxdb` tap by the release workflow.
#
# After publishing the tap, end users install with:
#   brew tap bankyresearch/duxxdb
#   brew install duxxdb
#   brew services start duxxdb
#
# Until the tap is published, install from this formula directly:
#   brew install --build-from-source ./packaging/homebrew/duxxdb.rb
#
# The placeholders (VERSION, SHA256_*) are filled in automatically by
# the GitHub Actions release job from the just-built archives.

class Duxxdb < Formula
  desc     "Database for AI agents (vector + BM25 + structured, RESP/gRPC/MCP)"
  homepage "https://github.com/bankyresearch/duxxdb"
  version  "0.1.0"
  license  "Apache-2.0"

  if Hardware::CPU.arm?
    url    "https://github.com/bankyresearch/duxxdb/releases/download/v#{version}/duxxdb-v#{version}-aarch64-macos.tar.gz"
    sha256 "REPLACE_WITH_AARCH64_MACOS_SHA256"
  else
    url    "https://github.com/bankyresearch/duxxdb/releases/download/v#{version}/duxxdb-v#{version}-x86_64-macos.tar.gz"
    sha256 "REPLACE_WITH_X86_64_MACOS_SHA256"
  end

  def install
    bin.install "duxx-server"
    bin.install "duxx-grpc"
    bin.install "duxx-mcp"
    bin.install "duxx-export"

    # Default config + data dirs.
    (etc/"duxxdb").mkpath
    (var/"lib/duxxdb").mkpath

    # Drop a starter env file if one isn't present yet.
    env_path = etc/"duxxdb/duxx.env"
    unless env_path.exist?
      env_path.write <<~EOS
        # /usr/local/etc/duxxdb/duxx.env (or /opt/homebrew/etc/duxxdb/duxx.env on Apple Silicon)
        # Generate a token:  openssl rand -hex 32
        DUXX_TOKEN=
        DUXX_STORAGE=dir:#{var}/lib/duxxdb
        DUXX_EMBEDDER=hash:32
        DUXX_METRICS_ADDR=127.0.0.1:9100
        RUST_LOG=info
      EOS
    end
  end

  service do
    run [opt_bin/"duxx-server", "--addr", "127.0.0.1:6379", "--drain-secs", "30"]
    environment_variables(
      DUXX_STORAGE: "dir:#{var}/lib/duxxdb",
      RUST_LOG:     "info"
    )
    keep_alive    true
    log_path      var/"log/duxxdb.log"
    error_log_path var/"log/duxxdb.err.log"
    working_dir   var/"lib/duxxdb"
  end

  test do
    output = shell_output("#{bin}/duxx-server --help")
    assert_match "USAGE: duxx-server", output
  end
end
