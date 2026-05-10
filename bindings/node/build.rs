// napi-rs needs a tiny build script to set up the Node-compatible
// linker flags (so the resulting .dll/.so is loadable by Node's
// process.dlopen).

extern crate napi_build;

fn main() {
    napi_build::setup();
}
