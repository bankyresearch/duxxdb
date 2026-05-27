//! Offline snapshot/verify/restore utility for DuxxDB dir storage.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MANIFEST_FILE: &str = "manifest.json";
const DATA_DIR: &str = "data";
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotManifest {
    schema_version: u32,
    duxx_version: String,
    created_at_unix_secs: u64,
    source: String,
    file_count: usize,
    total_bytes: u64,
    files: Vec<FileManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileManifest {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug)]
struct CreateArgs {
    source: PathBuf,
    out: PathBuf,
}

#[derive(Debug)]
struct VerifyArgs {
    snapshot: PathBuf,
}

#[derive(Debug)]
struct RestoreArgs {
    snapshot: PathBuf,
    target: PathBuf,
    force: bool,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };
    let rest: Vec<String> = args.collect();

    match command.as_str() {
        "create" => {
            if wants_help(&rest) {
                print_create_help();
                return Ok(());
            }
            let args = parse_create(&rest)?;
            let manifest = create_snapshot(&args.source, &args.out)?;
            println!(
                "created snapshot {} ({} files, {} bytes)",
                args.out.display(),
                manifest.file_count,
                manifest.total_bytes
            );
        }
        "verify" => {
            if wants_help(&rest) {
                print_verify_help();
                return Ok(());
            }
            let args = parse_verify(&rest)?;
            let manifest = verify_snapshot(&args.snapshot)?;
            println!(
                "verified snapshot {} ({} files, {} bytes)",
                args.snapshot.display(),
                manifest.file_count,
                manifest.total_bytes
            );
        }
        "restore" => {
            if wants_help(&rest) {
                print_restore_help();
                return Ok(());
            }
            let args = parse_restore(&rest)?;
            restore_snapshot(&args.snapshot, &args.target, args.force)?;
            println!(
                "restored snapshot {} -> {}",
                args.snapshot.display(),
                args.target.display()
            );
        }
        "--help" | "-h" => print_help(),
        other => anyhow::bail!("unknown command: {other}"),
    }

    Ok(())
}

fn parse_create(args: &[String]) -> anyhow::Result<CreateArgs> {
    let mut source = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source" | "-s" => {
                i += 1;
                source = Some(storage_dir_arg(args.get(i), "--source")?);
            }
            "--out" | "-o" => {
                i += 1;
                out = Some(path_arg(args.get(i), "--out")?);
            }
            other => anyhow::bail!("unknown create arg: {other}"),
        }
        i += 1;
    }
    Ok(CreateArgs {
        source: source.ok_or_else(|| anyhow::anyhow!("create requires --source dir:<path>"))?,
        out: out.ok_or_else(|| anyhow::anyhow!("create requires --out <snapshot-dir>"))?,
    })
}

fn parse_verify(args: &[String]) -> anyhow::Result<VerifyArgs> {
    let mut snapshot = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snapshot" => {
                i += 1;
                snapshot = Some(path_arg(args.get(i), "--snapshot")?);
            }
            other => anyhow::bail!("unknown verify arg: {other}"),
        }
        i += 1;
    }
    Ok(VerifyArgs {
        snapshot: snapshot.ok_or_else(|| anyhow::anyhow!("verify requires --snapshot <dir>"))?,
    })
}

fn parse_restore(args: &[String]) -> anyhow::Result<RestoreArgs> {
    let mut snapshot = None;
    let mut target = None;
    let mut force = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--snapshot" => {
                i += 1;
                snapshot = Some(path_arg(args.get(i), "--snapshot")?);
            }
            "--target" | "-t" => {
                i += 1;
                target = Some(storage_dir_arg(args.get(i), "--target")?);
            }
            "--force" => force = true,
            other => anyhow::bail!("unknown restore arg: {other}"),
        }
        i += 1;
    }
    Ok(RestoreArgs {
        snapshot: snapshot.ok_or_else(|| anyhow::anyhow!("restore requires --snapshot <dir>"))?,
        target: target.ok_or_else(|| anyhow::anyhow!("restore requires --target dir:<path>"))?,
        force,
    })
}

fn wants_help(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--help" || arg == "-h")
}

fn path_arg(value: Option<&String>, flag: &str) -> anyhow::Result<PathBuf> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a path"))
}

fn storage_dir_arg(value: Option<&String>, flag: &str) -> anyhow::Result<PathBuf> {
    let value = value.ok_or_else(|| anyhow::anyhow!("{flag} needs dir:<path> or <path>"))?;
    let path = value.strip_prefix("dir:").unwrap_or(value);
    if path.trim().is_empty() || path == "memory" {
        anyhow::bail!("{flag} must point at a persistent directory");
    }
    Ok(PathBuf::from(path))
}

fn create_snapshot(source: &Path, out: &Path) -> anyhow::Result<SnapshotManifest> {
    if !source.is_dir() {
        anyhow::bail!("source is not a directory: {}", source.display());
    }
    if out.exists() && !is_empty_dir(out)? {
        anyhow::bail!(
            "snapshot output already exists and is not empty: {}",
            out.display()
        );
    }

    let source_abs = canonical_or_absolute(source)?;
    let out_abs = canonical_or_absolute(out)?;
    if out_abs.starts_with(&source_abs) {
        anyhow::bail!("snapshot output must not be inside the source directory");
    }

    fs::create_dir_all(out)?;
    let data_root = out.join(DATA_DIR);
    if data_root.exists() && !is_empty_dir(&data_root)? {
        anyhow::bail!("snapshot data directory already exists and is not empty");
    }
    fs::create_dir_all(&data_root)?;

    let files = copy_source_files(source, &data_root)?;
    let total_bytes = files.iter().map(|f| f.bytes).sum();
    let manifest = SnapshotManifest {
        schema_version: SCHEMA_VERSION,
        duxx_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_unix_secs: unix_now()?,
        source: source.display().to_string(),
        file_count: files.len(),
        total_bytes,
        files,
    };

    let body = serde_json::to_vec_pretty(&manifest)?;
    let manifest_path = out.join(MANIFEST_FILE);
    let mut f = File::create(&manifest_path)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", manifest_path.display()))?;
    f.write_all(&body)?;
    f.write_all(b"\n")?;
    f.sync_all().ok();
    Ok(manifest)
}

fn verify_snapshot(snapshot: &Path) -> anyhow::Result<SnapshotManifest> {
    let manifest = read_manifest(snapshot)?;
    if manifest.schema_version != SCHEMA_VERSION {
        anyhow::bail!(
            "unsupported snapshot schema {} (expected {})",
            manifest.schema_version,
            SCHEMA_VERSION
        );
    }

    let data_root = snapshot.join(DATA_DIR);
    if !data_root.is_dir() {
        anyhow::bail!("missing snapshot data directory: {}", data_root.display());
    }

    let mut expected = HashSet::new();
    let mut total_bytes = 0u64;
    for file in &manifest.files {
        if !expected.insert(file.path.clone()) {
            anyhow::bail!("duplicate file in manifest: {}", file.path);
        }
        let path = safe_join(&data_root, &file.path)?;
        let meta = fs::metadata(&path)
            .map_err(|e| anyhow::anyhow!("snapshot file missing {}: {e}", file.path))?;
        if !meta.is_file() {
            anyhow::bail!("snapshot entry is not a file: {}", file.path);
        }
        if meta.len() != file.bytes {
            anyhow::bail!(
                "size mismatch for {}: manifest={} actual={}",
                file.path,
                file.bytes,
                meta.len()
            );
        }
        let got = sha256_file(&path)?;
        if got != file.sha256 {
            anyhow::bail!("sha256 mismatch for {}", file.path);
        }
        total_bytes += file.bytes;
    }

    let actual = collect_manifest_paths(&data_root)?;
    for path in actual {
        if !expected.contains(&path) {
            anyhow::bail!("unexpected file in snapshot data: {path}");
        }
    }

    if manifest.file_count != manifest.files.len() {
        anyhow::bail!(
            "manifest file_count mismatch: {} != {}",
            manifest.file_count,
            manifest.files.len()
        );
    }
    if manifest.total_bytes != total_bytes {
        anyhow::bail!(
            "manifest total_bytes mismatch: {} != {}",
            manifest.total_bytes,
            total_bytes
        );
    }

    Ok(manifest)
}

fn restore_snapshot(snapshot: &Path, target: &Path, force: bool) -> anyhow::Result<()> {
    verify_snapshot(snapshot)?;

    if target.exists() {
        if !target.is_dir() {
            anyhow::bail!(
                "restore target exists and is not a directory: {}",
                target.display()
            );
        }
        if !is_empty_dir(target)? {
            if !force {
                anyhow::bail!("restore target is not empty; pass --force to move it aside first");
            }
            let backup = backup_path(target)?;
            fs::rename(target, &backup).map_err(|e| {
                anyhow::anyhow!(
                    "move existing target {} to {}: {e}",
                    target.display(),
                    backup.display()
                )
            })?;
        }
    }

    fs::create_dir_all(target)?;
    copy_snapshot_data(&snapshot.join(DATA_DIR), target)?;
    Ok(())
}

fn read_manifest(snapshot: &Path) -> anyhow::Result<SnapshotManifest> {
    let manifest_path = snapshot.join(MANIFEST_FILE);
    let body = fs::read(&manifest_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", manifest_path.display()))?;
    Ok(serde_json::from_slice(&body)?)
}

fn copy_source_files(source: &Path, data_root: &Path) -> anyhow::Result<Vec<FileManifest>> {
    let mut files = collect_files(source)?;
    files.sort();

    let mut manifest = Vec::with_capacity(files.len());
    for src in files {
        let rel = relative_manifest_path(source, &src)?;
        let dst = safe_join(data_root, &rel)?;
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)
            .map_err(|e| anyhow::anyhow!("copy {} -> {}: {e}", src.display(), dst.display()))?;
        let meta = fs::metadata(&dst)?;
        manifest.push(FileManifest {
            path: rel,
            bytes: meta.len(),
            sha256: sha256_file(&dst)?,
        });
    }
    Ok(manifest)
}

fn copy_snapshot_data(data_root: &Path, target: &Path) -> anyhow::Result<()> {
    for src in collect_files(data_root)? {
        let rel = relative_manifest_path(data_root, &src)?;
        let dst = safe_join(target, &rel)?;
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst)
            .map_err(|e| anyhow::anyhow!("copy {} -> {}: {e}", src.display(), dst.display()))?;
    }
    Ok(())
}

fn collect_manifest_paths(root: &Path) -> anyhow::Result<HashSet<String>> {
    collect_files(root)?
        .into_iter()
        .map(|path| relative_manifest_path(root, &path))
        .collect()
}

fn collect_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_inner(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_inner(dir: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in
        fs::read_dir(dir).map_err(|e| anyhow::anyhow!("read dir {}: {e}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files_inner(&path, files)?;
        } else if file_type.is_file() {
            files.push(path);
        } else {
            anyhow::bail!(
                "unsupported non-file path in snapshot source: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn relative_manifest_path(root: &Path, path: &Path) -> anyhow::Result<String> {
    let rel = path.strip_prefix(root)?;
    let mut parts = Vec::new();
    for component in rel.components() {
        match component {
            Component::Normal(part) => {
                let s = part
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("snapshot paths must be UTF-8"))?;
                parts.push(s.to_string());
            }
            _ => anyhow::bail!("invalid relative snapshot path: {}", rel.display()),
        }
    }
    Ok(parts.join("/"))
}

fn safe_join(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    if rel.is_empty() {
        anyhow::bail!("empty snapshot path");
    }
    let rel_path = Path::new(rel);
    let mut out = root.to_path_buf();
    for component in rel_path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            _ => anyhow::bail!("unsafe snapshot path: {rel}"),
        }
    }
    Ok(out)
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f = File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn is_empty_dir(path: &Path) -> anyhow::Result<bool> {
    if !path.exists() {
        return Ok(true);
    }
    if !path.is_dir() {
        anyhow::bail!("not a directory: {}", path.display());
    }
    Ok(fs::read_dir(path)?.next().is_none())
}

fn canonical_or_absolute(path: &Path) -> anyhow::Result<PathBuf> {
    if path.exists() {
        return Ok(path.canonicalize()?);
    }

    let mut missing: Vec<OsString> = Vec::new();
    let mut cursor = path;
    loop {
        if cursor.exists() {
            let mut out = cursor.canonicalize()?;
            for part in missing.iter().rev() {
                out.push(part);
            }
            return Ok(out);
        }

        let Some(name) = cursor.file_name() else {
            let mut out = std::env::current_dir()?.canonicalize()?;
            for part in missing.iter().rev() {
                out.push(part);
            }
            return Ok(out);
        };
        missing.push(name.to_os_string());

        let Some(parent) = cursor.parent().filter(|p| !p.as_os_str().is_empty()) else {
            let mut out = std::env::current_dir()?.canonicalize()?;
            for part in missing.iter().rev() {
                out.push(part);
            }
            return Ok(out);
        };
        cursor = parent;
    }
}

fn backup_path(target: &Path) -> anyhow::Result<PathBuf> {
    let ts = unix_now()?;
    let name = target
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("target has no file name: {}", target.display()))?;
    let backup_name = format!("{name}.pre-restore-{ts}");
    Ok(target.with_file_name(backup_name))
}

fn unix_now() -> anyhow::Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn print_help() {
    println!("duxx-snapshot v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("USAGE:");
    println!("  duxx-snapshot create  --source dir:<storage-dir> --out <snapshot-dir>");
    println!("  duxx-snapshot verify  --snapshot <snapshot-dir>");
    println!(
        "  duxx-snapshot restore --snapshot <snapshot-dir> --target dir:<storage-dir> [--force]"
    );
    println!();
    println!("Stop duxx-server or otherwise quiesce writes before creating a snapshot.");
}

fn print_create_help() {
    println!("USAGE: duxx-snapshot create --source dir:<storage-dir> --out <snapshot-dir>");
}

fn print_verify_help() {
    println!("USAGE: duxx-snapshot verify --snapshot <snapshot-dir>");
}

fn print_restore_help() {
    println!("USAGE: duxx-snapshot restore --snapshot <snapshot-dir> --target dir:<storage-dir> [--force]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_verify_restore_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        fs::create_dir_all(source.join("phase7")).unwrap();
        fs::write(source.join("memory.redb"), b"memory").unwrap();
        fs::write(source.join("phase7").join("traces.redb"), b"traces").unwrap();

        let snapshot = tmp.path().join("snapshot");
        let manifest = create_snapshot(&source, &snapshot).unwrap();
        assert_eq!(manifest.file_count, 2);
        verify_snapshot(&snapshot).unwrap();

        let target = tmp.path().join("restore");
        restore_snapshot(&snapshot, &target, false).unwrap();
        assert_eq!(fs::read(target.join("memory.redb")).unwrap(), b"memory");
        assert_eq!(
            fs::read(target.join("phase7").join("traces.redb")).unwrap(),
            b"traces"
        );
    }

    #[test]
    fn verify_rejects_tampered_file() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("memory.redb"), b"memory").unwrap();

        let snapshot = tmp.path().join("snapshot");
        create_snapshot(&source, &snapshot).unwrap();
        fs::write(snapshot.join(DATA_DIR).join("memory.redb"), b"tampered").unwrap();

        let err = verify_snapshot(&snapshot).unwrap_err().to_string();
        assert!(err.contains("size mismatch") || err.contains("sha256 mismatch"));
    }

    #[test]
    fn restore_moves_existing_target_when_forced() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("memory.redb"), b"memory").unwrap();
        let snapshot = tmp.path().join("snapshot");
        create_snapshot(&source, &snapshot).unwrap();

        let target = tmp.path().join("target");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("old.redb"), b"old").unwrap();

        restore_snapshot(&snapshot, &target, true).unwrap();
        assert_eq!(fs::read(target.join("memory.redb")).unwrap(), b"memory");
        let backups: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| name.starts_with("target.pre-restore-"))
            .collect();
        assert_eq!(backups.len(), 1);
    }
}
