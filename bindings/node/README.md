# duxxdb — Node.js / TypeScript bindings

Native bindings for [DuxxDB](https://github.com/duxxdb/duxxdb), the
agent-native hybrid database. Built with [napi-rs] v2 — Rust core
compiled to a native `.node` module + auto-generated TypeScript
declarations.

## Install (after build)

```bash
cd bindings/node
npm install                # pulls @napi-rs/cli
npm run build              # produces *.node + index.js + index.d.ts
npm test                   # smoke-tests against the built module
```

Locally: `npm link` then `npm link duxxdb` from your project.
npm publish lands with v0.1.0.

### Build platforms

| Platform | Status |
|---|---|
| **Linux x86_64** (any toolchain) | ✅ `npm run build` works out of the box |
| **macOS x86_64 / arm64** | ✅ `npm run build` works out of the box |
| **Windows MSVC** | ✅ Works once the Windows SDK is installed; `napi-build` reads symbols from `node.lib`. |
| **Windows mingw / Rust GNU** | ⚠ Blocked: `napi-build` needs `libnode.dll.a`, which the rustup-bundled mingw toolchain doesn't ship. Use the MSVC toolchain on Windows (`rustup default stable-x86_64-pc-windows-msvc`) or build in CI on Linux. |

The Rust code is identical across platforms — only the linker step
differs. CI is the recommended build path; per-platform `.node`
binaries are then published to npm.

## Quickstart

```ts
import { MemoryStore, ToolCache, SessionStore, version } from "duxxdb";

console.log(`duxxdb v${version()}`);

const store = new MemoryStore(4);

function embed(text: string): number[] {
    // Replace with a real embedder (OpenAI, Cohere, local BGE).
    const v = [0, 0, 0, 0];
    for (const t of text.toLowerCase().split(/\s+/)) {
        let h = 0;
        for (const c of t) h = (h * 31 + c.charCodeAt(0)) | 0;
        v[((h % 4) + 4) % 4] += 1;
    }
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map(x => x / n);
}

store.remember("alice", "I lost my wallet at the cafe", embed("wallet"));
store.remember("alice", "Favorite color is blue", embed("blue"));

const hits = store.recall("alice", "wallet", embed("wallet"), 5);
for (const h of hits) console.log(h.score.toFixed(4), h.text);
```

## API

| Class | Constructor | Methods |
|---|---|---|
| `MemoryStore` | `(dim: number, capacity?: number)` | `remember(key, text, embedding) -> id`, `recall(key, query, embedding, k?) -> MemoryHit[]`, getters: `dim`, `len` |
| `MemoryHit` | _(returned)_ | `id`, `key`, `text`, `score` |
| `ToolCache` | `(threshold?: number = 0.95)` | `put(tool, argsHash, argsEmbedding, result: Buffer, ttlSecs?)`, `get(...) -> ToolCacheHit \| null`, `purgeExpired()`, getter `len` |
| `ToolCacheHit` | _(returned)_ | `kind: "exact" \| "semantic_near_hit"`, `similarity`, `result: Buffer` |
| `SessionStore` | `(ttlSecs?: number = 1800)` | `put(sessionId, data: Buffer)`, `get(sessionId) -> Buffer \| null`, `delete(...) -> boolean`, `purgeExpired()`, getter `len` |

Module-level: `version(): string`.

## Why napi-rs?

- N-API is the stable Node ABI — one binary works across many Node
  versions.
- napi-rs v2 supports Buffer, BigInt, async tasks, structured types
  out of the box.
- Bun also speaks N-API; the same `.node` module loads there.

## License

Apache 2.0. See [../../LICENSE](../../LICENSE).

[napi-rs]: https://github.com/napi-rs/napi-rs
