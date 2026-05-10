// End-to-end smoke test for the duxxdb Node.js bindings.
// Run after `npm run build` produces index.js / index.d.ts / *.node.

const assert = require("node:assert");
const { MemoryStore, ToolCache, SessionStore, version } = require("./index.js");

console.log(`duxxdb version: ${version()}`);

// MemoryStore
const store = new MemoryStore(4);
assert.strictEqual(store.dim, 4);

// Toy embedder: simple hash bucketing.
function embed(text) {
    const v = [0, 0, 0, 0];
    for (const tok of text.toLowerCase().split(/\s+/)) {
        let h = 0;
        for (let i = 0; i < tok.length; i++) {
            h = (h * 31 + tok.charCodeAt(i)) | 0;
        }
        v[((h % 4) + 4) % 4] += 1;
    }
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map(x => x / norm);
}

const id1 = store.remember("alice", "I lost my wallet at the cafe", embed("wallet"));
const id2 = store.remember("alice", "My favorite color is blue", embed("blue"));
console.log(`stored memories: ${id1}, ${id2}`);
assert.strictEqual(store.len, 2);

const hits = store.recall("alice", "wallet", embed("wallet"), 2);
console.log("recall hits:");
for (const h of hits) {
    console.log(`  id=${h.id} score=${h.score.toFixed(4)} text=${JSON.stringify(h.text)}`);
}
assert.ok(hits.length > 0, "expected at least one hit");
assert.ok(hits[0].text.includes("wallet"), "top hit should mention wallet");

// ToolCache
const cache = new ToolCache(0.95);
cache.put("web_search", 42, [1, 0, 0, 0], Buffer.from("answer"), 60);
const exact = cache.get("web_search", 42, [1, 0, 0, 0]);
console.log("exact hit:", exact);
assert.strictEqual(exact.kind, "exact");
assert.strictEqual(exact.result.toString(), "answer");

// Different hash, very similar embedding -> semantic near hit.
const near = cache.get("web_search", 999, [0.99, 0.05, 0, 0]);
console.log("near hit:", near);
assert.strictEqual(near.kind, "semantic_near_hit");
assert.ok(near.similarity >= 0.95);

// Orthogonal embedding -> miss.
const miss = cache.get("web_search", 100, [0, 0, 1, 0]);
assert.strictEqual(miss, null);
console.log("miss case OK");

// SessionStore
const sess = new SessionStore(60);
sess.put("sid-1", Buffer.from("session blob"));
const got = sess.get("sid-1");
assert.strictEqual(got.toString(), "session blob");
console.log("session round-trip OK");

console.log("\nALL TESTS PASS");
