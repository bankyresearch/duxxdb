# DuxxDB — Setup Log

Running log of environment setup for the DuxxDB project. Every command and decision is recorded so any contributor can reproduce the environment from scratch.

---

## Target environment

- **OS:** Windows 11 (primary dev machine). Linux and macOS support follow the same toolchain.
- **Shell:** Git Bash (MSYS2) for commands. PowerShell also works.
- **Working directory:** `D:\DuxxDB`

---

## Step 1 — Prerequisites check

Ran on 2026-04-23:

```bash
rustc --version   # ❌ not found
cargo --version   # ❌ not found
git --version     # ✅ git version 2.45.1.windows.1
```

**Conclusion:** Rust toolchain missing. Git present. Proceed to install Rust.

---

## Step 2 — Install Rust via winget

Chosen installation method: **winget** (Windows Package Manager, pre-installed on Win 10/11).

Why winget over `rustup-init.exe` manual download:
- Scriptable, unattended.
- Same command works on any Win 10/11 machine.
- Handles PATH setup automatically.

### Command

```bash
winget install Rustlang.Rustup \
  --accept-source-agreements \
  --accept-package-agreements \
  --silent
```

Flags explained:
- `--accept-source-agreements` — skips prompt for winget repo terms.
- `--accept-package-agreements` — skips prompt for package license terms.
- `--silent` — no interactive installer UI.

### What this installs

- **rustup** — the Rust toolchain manager.
- **stable toolchain** (latest stable: rustc, cargo, rustfmt, clippy).
- **MSVC target** (`x86_64-pc-windows-msvc`) — default on Windows.

### Secondary requirement: Visual Studio Build Tools

Rust on Windows with the MSVC toolchain needs the **VS Build Tools C++ workload** to link native deps (Lance, usearch).

If missing, rustup prints a link during install. Installed separately via:

```bash
winget install Microsoft.VisualStudio.2022.BuildTools \
  --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

(Only run if rustup reports missing C++ tools.)

---

## Step 3 — Verify install

After install, open a **fresh shell** (so PATH picks up `~/.cargo/bin`):

```bash
rustc --version
cargo --version
rustup --version
rustup show
```

Expected: rustc ≥ 1.75, cargo matching version, active toolchain = `stable-x86_64-pc-windows-msvc`.

### Actual result (2026-04-23)

```
rustc 1.95.0 (59807616e 2026-04-14)
cargo 1.95.0 (f2d3ce0bd 2026-03-21)
rustup 1.29.0 (28d1352db 2026-03-05)
```

Binaries installed at `C:\Users\ibkc\.cargo\bin\`:
```
cargo.exe, cargo-clippy.exe, cargo-fmt.exe, cargo-miri.exe,
clippy-driver.exe, rls.exe, rust-analyzer.exe, rust-gdb.exe,
rust-gdbgui.exe, rust-lldb.exe, rustc.exe, rustdoc.exe,
rustfmt.exe, rustup.exe
```

Note: within a non-interactive shell that pre-dates the install, prefix `PATH`:
```bash
export PATH="/c/Users/ibkc/.cargo/bin:$PATH"
```
New shells after the install do not need this — rustup updates the user PATH.

---

## Step 4 — Git repo init

Per user decision: **local-only init** for now (no GitHub push yet).

```bash
cd /d/DuxxDB
git init
git config user.name "<your name>"
git config user.email "<your email>"
```

`.gitignore` covers:
- `target/` (Cargo build output)
- `Cargo.lock` is committed (this is an application/binary workspace, not a library; committing lock is correct).
- `*.redb`, `*.lance`, `data/` (runtime data).
- IDE files (`.vscode/`, `.idea/`).

---

## Step 5 — Workspace scaffold

After Rust is verified, the Cargo workspace is created with **10 crates**. See [ROADMAP.md](./ROADMAP.md) for the exact scaffold order and [ARCHITECTURE.md](./ARCHITECTURE.md) for what each crate does.

---

## Step 6 — First build smoke test

```bash
cd /d/DuxxDB
cargo build --workspace
cargo test --workspace
cargo run --example chatbot_memory
```

First build pulls ~200 crates and takes 3–8 min on a modern laptop. Subsequent builds are incremental (seconds).

---

## Step 6 bis — Windows toolchain choice (the real story)

Rust on Windows has two linker backends:

- **`stable-x86_64-pc-windows-msvc`** (rustup's default) — uses MSVC `link.exe`.
  Needs Visual Studio Build Tools **with** the Windows SDK component
  installed AND a shell environment set up by `vcvars64.bat` so `LIB` /
  `INCLUDE` resolve. Without those, linking fails with
  `LNK1181: cannot open input file 'kernel32.lib'`.
- **`stable-x86_64-pc-windows-gnu`** — uses MinGW-w64 `ld.exe`. rustup
  ships a partial "self-contained" mingw, but `dlltool.exe` in there
  needs `gcc`/`as` from a real MinGW install. Without those, you get
  `dlltool: CreateProcess` errors.

### What actually worked on this machine

The fastest unblock for a Windows + Git Bash dev environment **without
admin elevation**:

1. Install Rust (sets up rustup default = MSVC):
   ```bash
   winget install Rustlang.Rustup --silent
   ```
2. Add the GNU toolchain (avoids MSVC + SDK requirement):
   ```bash
   rustup toolchain install stable-x86_64-pc-windows-gnu
   rustup default stable-x86_64-pc-windows-gnu
   ```
3. Install **WinLibs MinGW (POSIX threads, MSVCRT runtime)** — provides
   `gcc`, `as`, `ld`, `ar`, `dlltool` that the GNU toolchain needs:
   ```bash
   winget install BrechtSanders.WinLibs.POSIX.MSVCRT --silent \
     --accept-package-agreements --accept-source-agreements
   ```
4. Prepend the WinLibs `bin/` to PATH and build:
   ```bash
   MINGW_BIN="/c/Users/<you>/AppData/Local/Microsoft/WinGet/Packages/\
BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
   export PATH="$MINGW_BIN:/c/Users/<you>/.cargo/bin:$PATH"
   cargo test --workspace
   ```

   For convenience, `scripts/build.sh` does the PATH setup for you.

### Failure modes we hit and resolved (in order)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `link: extra operand` | Git Bash `/usr/bin/link` (GNU coreutils, makes hard links) shadows MSVC `link.exe` | Switch to GNU toolchain (avoids `link.exe` entirely) |
| 2 | `LNK1181: cannot open input file 'kernel32.lib'` | MSVC `link.exe` can't find the Windows SDK because the SDK component isn't installed | Either install the SDK (needs admin), or switch to GNU |
| 3 | `error calling dlltool 'dlltool.exe': program not found` | rustup's GNU toolchain expects `dlltool` on PATH; it's not there | Install a real MinGW (we used WinLibs MSVCRT) |
| 4 | `dlltool: CreateProcess` (after step 3, before WinLibs) | rustup's bundled `self-contained/dlltool.exe` tries to spawn `gcc`/`as` and they aren't there | Same — install full MinGW |
| 5 | `Commands with --quiet or --passive should be run elevated` | VS Installer can't modify a system install without admin | Don't go down the MSVC path until SDK is installed via an admin shell or GUI |

### Switching to MSVC later (when you want it)

Phase 2+ may want MSVC for Lance / usearch's native bits. Once you can
install the SDK with admin:

```powershell
# In an elevated PowerShell:
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" `
  modify --installPath "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools" `
  --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --quiet --norestart
```

Then in your shell:

```bash
rustup default stable-x86_64-pc-windows-msvc
```

…and use `scripts/build.bat` (which sources `vcvars64.bat`) to build.

---

## Troubleshooting reference

| Symptom | Fix |
|---|---|
| `cargo: command not found` after install | Close and reopen terminal (PATH not reloaded). |
| `link: extra operand` on Windows | MSYS2 shadowing MSVC linker — use `scripts/build.sh` (see Step 6 bis). |
| `error: linker 'cc' not found` | Need MSVC linker — install VS 2022 Build Tools C++ workload. |
| Lance/usearch build fails with CMake error | Install CMake: `winget install Kitware.CMake`. |
| Slow first build | Normal — ~80 deps in Phase 1, ~300 once Lance lands. Use `sccache` for future speedups. |

---

## Provenance

Every change to this file should accompany an actual change to the environment. This is not an install guide in the abstract — it's a **log of what was done**.
