# One-command reproducer for the DuxxDB comparative benchmark (Windows).
#
#   ./run.ps1 -Quick              # embedded DuxxDB + exact reference; no services
#   ./run.ps1                     # full suite: redis/qdrant/pgvector + duxx-grpc
#   ./run.ps1 -Args "--dims 128,768 --n 50000 --queries 500"
#
# Results print as markdown and land in results-<timestamp>.json.
# See docs/BENCHMARKS.md for methodology and hardware-reporting rules.
param(
  [switch]$Quick,
  [string]$Args = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$root = (Resolve-Path "../..").Path
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$out = "results-$ts.json"
$extra = if ($Args) { $Args.Split(" ") } else { @() }

try { python -c "import numpy" 2>$null } catch { pip install -q numpy }

if ($Quick) {
  Write-Host ">> quick mode: embedded DuxxDB + reference only"
  python bench.py --quick @extra --out $out
  exit $LASTEXITCODE
}

Write-Host ">> starting redis / qdrant / pgvector via docker compose"
docker compose up -d redis-stack qdrant pgvector

Write-Host ">> waiting for services"
foreach ($i in 1..60) {
  $ready = $true
  foreach ($p in 6379, 6333, 5432) {
    try { (New-Object Net.Sockets.TcpClient("localhost", $p)).Close() }
    catch { $ready = $false }
  }
  if ($ready) { break }
  Start-Sleep 2
}

Write-Host ">> building duxx-grpc (release)"
Push-Location $root; cargo build --release -p duxx-grpc; Pop-Location
$data = New-Item -ItemType Directory -Path (Join-Path $env:TEMP "duxx-bench-$ts")
Write-Host ">> launching duxx-grpc on 127.0.0.1:50051"
$daemon = Start-Process -PassThru -FilePath "$root/target/release/duxx-grpc.exe" `
  -ArgumentList "--addr", "127.0.0.1:50051", "--storage", "dir:$data"
Start-Sleep 3

pip install -q redis qdrant-client psycopg2-binary grpcio 2>$null

try {
  Write-Host ">> running benchmark"
  python bench.py @extra --out $out
  Write-Host ">> wrote $out"
} finally {
  Write-Host ">> tearing down"
  if ($daemon -and -not $daemon.HasExited) { $daemon.Kill() }
  docker compose down -v
  Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue
}
