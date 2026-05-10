// Build-time: tonic-build invokes `protoc` to generate Rust types from
// proto/duxx.proto into `OUT_DIR/duxx.v1.rs`. The lib.rs `pb!` module
// includes that file via `include!`.
//
// Requires `protoc` on PATH at build time. On Windows:
//   winget install Google.Protobuf
// On macOS: `brew install protobuf`. On Debian: `apt install protobuf-compiler`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/duxx.proto"], &["proto"])?;
    Ok(())
}
