//! # duxxdb-node
//!
//! Node.js / TypeScript bindings for DuxxDB. Built with napi-rs v2.
//! `npm run build` produces a `.node` native module + an `index.js`
//! shim + an `index.d.ts` typings file.
//!
//! ## Public JS surface
//!
//! ```ts
//! import { MemoryStore, ToolCache, SessionStore } from "duxxdb";
//!
//! const store = new MemoryStore(4);
//! store.remember("alice", "hello world", [1, 0, 0, 0]);
//! const hits = store.recall("alice", "hello", [1, 0, 0, 0], 5);
//! ```

#![deny(clippy::all)]

use duxx_memory::{
    HitKind as RustHitKind, MemoryStore as RustMemoryStore, SessionStore as RustSessionStore,
    ToolCache as RustToolCache,
};
use napi::bindgen_prelude::Buffer;
use napi::Error as NapiError;
use napi_derive::napi;
use std::time::Duration;

fn napi_err(e: impl std::fmt::Display) -> NapiError {
    NapiError::from_reason(e.to_string())
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

#[napi]
pub struct MemoryStore {
    inner: RustMemoryStore,
}

#[napi]
impl MemoryStore {
    /// Construct a store. `capacity` defaults to 100_000.
    #[napi(constructor)]
    pub fn new(dim: u32, capacity: Option<u32>) -> Self {
        Self {
            inner: RustMemoryStore::with_capacity(
                dim as usize,
                capacity.unwrap_or(100_000) as usize,
            ),
        }
    }

    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }

    /// Number of stored memories.
    #[napi(getter)]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Insert a memory; returns the assigned id.
    #[napi]
    pub fn remember(
        &self,
        key: String,
        text: String,
        embedding: Vec<f64>,
    ) -> napi::Result<u32> {
        let emb: Vec<f32> = embedding.into_iter().map(|x| x as f32).collect();
        if emb.len() != self.inner.dim() {
            return Err(napi_err(format!(
                "embedding has dim {}, store expects {}",
                emb.len(),
                self.inner.dim()
            )));
        }
        let id = self.inner.remember(key, text, emb).map_err(napi_err)?;
        Ok(id as u32)
    }

    /// Hybrid recall (vector + BM25, RRF-fused). `k` defaults to 10.
    #[napi]
    pub fn recall(
        &self,
        key: String,
        query: String,
        embedding: Vec<f64>,
        k: Option<u32>,
    ) -> napi::Result<Vec<MemoryHit>> {
        let qvec: Vec<f32> = embedding.into_iter().map(|x| x as f32).collect();
        if qvec.len() != self.inner.dim() {
            return Err(napi_err(format!(
                "embedding has dim {}, store expects {}",
                qvec.len(),
                self.inner.dim()
            )));
        }
        let hits = self
            .inner
            .recall(&key, &query, &qvec, k.unwrap_or(10) as usize)
            .map_err(napi_err)?;
        Ok(hits
            .into_iter()
            .map(|h| MemoryHit {
                id: h.memory.id as u32,
                key: h.memory.key,
                text: h.memory.text,
                score: h.score as f64,
            })
            .collect())
    }
}

#[napi(object)]
pub struct MemoryHit {
    pub id: u32,
    pub key: String,
    pub text: String,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// ToolCache
// ---------------------------------------------------------------------------

#[napi]
pub struct ToolCache {
    inner: RustToolCache,
}

#[napi]
impl ToolCache {
    /// Construct with a similarity threshold (default 0.95).
    #[napi(constructor)]
    pub fn new(threshold: Option<f64>) -> Self {
        Self {
            inner: RustToolCache::with_threshold(threshold.unwrap_or(0.95) as f32),
        }
    }

    #[napi(getter)]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Insert / overwrite a cache entry. `ttlSecs` defaults to 3600.
    #[napi]
    pub fn put(
        &self,
        tool: String,
        args_hash: u32,
        args_embedding: Vec<f64>,
        result: Buffer,
        ttl_secs: Option<u32>,
    ) -> napi::Result<()> {
        let emb: Vec<f32> = args_embedding.into_iter().map(|x| x as f32).collect();
        let bytes: Vec<u8> = result.to_vec();
        self.inner
            .put(
                tool,
                args_hash as u64,
                emb,
                bytes,
                Duration::from_secs(ttl_secs.unwrap_or(3600) as u64),
            )
            .map_err(napi_err)
    }

    /// Look up a cache entry. Returns `null` on miss.
    #[napi]
    pub fn get(
        &self,
        tool: String,
        args_hash: u32,
        args_embedding: Vec<f64>,
    ) -> Option<ToolCacheHit> {
        let emb: Vec<f32> = args_embedding.into_iter().map(|x| x as f32).collect();
        self.inner.get(&tool, args_hash as u64, &emb).map(|h| {
            ToolCacheHit {
                kind: match h.kind {
                    RustHitKind::Exact => "exact".to_string(),
                    RustHitKind::SemanticNearHit => "semantic_near_hit".to_string(),
                },
                similarity: h.similarity as f64,
                result: h.result.into(),
            }
        })
    }

    #[napi]
    pub fn purge_expired(&self) -> u32 {
        self.inner.purge_expired() as u32
    }
}

#[napi(object)]
pub struct ToolCacheHit {
    /// Either `"exact"` or `"semantic_near_hit"`.
    pub kind: String,
    pub similarity: f64,
    pub result: Buffer,
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

#[napi]
pub struct SessionStore {
    inner: RustSessionStore,
}

#[napi]
impl SessionStore {
    /// `ttlSecs` defaults to 1800 (30 min).
    #[napi(constructor)]
    pub fn new(ttl_secs: Option<u32>) -> Self {
        Self {
            inner: RustSessionStore::with_ttl(Duration::from_secs(
                ttl_secs.unwrap_or(1800) as u64,
            )),
        }
    }

    #[napi(getter)]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    #[napi]
    pub fn put(&self, session_id: String, data: Buffer) {
        self.inner.put(session_id, data.to_vec());
    }

    #[napi]
    pub fn get(&self, session_id: String) -> Option<Buffer> {
        self.inner.get(&session_id).map(Buffer::from)
    }

    #[napi]
    pub fn delete(&self, session_id: String) -> bool {
        self.inner.delete(&session_id)
    }

    #[napi]
    pub fn purge_expired(&self) -> u32 {
        self.inner.purge_expired() as u32
    }
}

// ---------------------------------------------------------------------------
// Module-level
// ---------------------------------------------------------------------------

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
