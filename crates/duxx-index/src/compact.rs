//! Compact HNSW: the same graph search as `hnsw_rs`, laid out for web-scale
//! corpora.
//!
//! `hnsw_rs` keeps every point as its own heap object with sixteen per-layer
//! neighbour vectors, and every neighbour as a separately allocated `Arc`.
//! At 128 dimensions that costs about 2.5 KB per point on top of the vector
//! itself; at four million web pages it was most of DuxxDB's memory.
//!
//! This index keeps:
//! - each vector as `dim` signed bytes plus one `f32` scale (vectors are
//!   unit-normalised first, so cosine similarity is a dot product);
//! - every point's layer-0 neighbours in one flat `u32` array, `M0` slots per
//!   point, `u32::MAX` marking an empty slot;
//! - the neighbours on upper layers (about one point in `M`) in a map from
//!   point to a flat slice of `M` slots per layer.
//!
//! About 270 bytes per 128-d point, roughly a tenth of `hnsw_rs`. Search is
//! the usual HNSW descent; the query stays `f32` and is scored against the
//! stored bytes (asymmetric distance), which keeps the order of the nearest
//! points close to exact `f32` cosine. Construction uses the neighbour
//! selection heuristic of Malkov and Yashunin (2016), filling any slots the
//! heuristic leaves empty with the nearest pruned candidates.
//!
//! The index has no dependencies beyond `std`, grows without a preset
//! capacity, and saves to and loads from a single file.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Neighbours per point on upper layers.
pub const M: usize = 16;
/// Neighbours per point on layer 0.
pub const M0: usize = 2 * M;
/// Highest layer a point can be given.
const MAX_LEVEL: u8 = 15;
const NONE: u32 = u32::MAX;
const MAGIC: &[u8; 8] = b"DUXXCHN1";

/// A candidate: a point and its similarity to the current query.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cand {
    sim: f32,
    id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.sim
            .partial_cmp(&o.sim)
            .unwrap_or(Ordering::Equal)
            .then(o.id.cmp(&self.id))
    }
}

/// Min-heap wrapper (the worst kept result on top).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Worst(std::cmp::Reverse<Cand>);

/// Small open-addressing set of visited points, sized to the search rather
/// than to the index (a per-index bitmap would cost bytes per point).
struct Visited {
    slots: Vec<u32>,
    len: usize,
}

impl Visited {
    fn new() -> Self {
        Self {
            slots: vec![NONE; 1024],
            len: 0,
        }
    }

    /// Insert `id`; false when it was already there.
    fn insert(&mut self, id: u32) -> bool {
        if (self.len + 1) * 2 > self.slots.len() {
            let grown = vec![NONE; self.slots.len() * 2];
            let old = std::mem::replace(&mut self.slots, grown);
            self.len = 0;
            for v in old.into_iter().filter(|v| *v != NONE) {
                self.insert(v);
            }
        }
        let mask = self.slots.len() - 1;
        let mut i = (id.wrapping_mul(0x9E37_79B9) as usize) & mask;
        loop {
            let s = self.slots[i];
            if s == NONE {
                self.slots[i] = id;
                self.len += 1;
                return true;
            }
            if s == id {
                return false;
            }
            i = (i + 1) & mask;
        }
    }
}

/// Unit-normalise `v` and quantise it to signed bytes with one scale.
/// Returns the normalised `f32` vector too (used while inserting).
fn quantise(v: &[f32]) -> (Vec<f32>, Vec<i8>, f32) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let unit: Vec<f32> = if norm > 0.0 && norm.is_finite() {
        v.iter().map(|x| x / norm).collect()
    } else {
        vec![0.0; v.len()]
    };
    let max = unit.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max == 0.0 {
        return (unit, vec![0; v.len()], 0.0);
    }
    let scale = max / 127.0;
    let codes = unit
        .iter()
        .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (unit, codes, scale)
}

fn dot_f_q(q: &[f32], codes: &[i8]) -> f32 {
    // Four accumulators let the compiler vectorise without fast-math.
    let mut acc = [0.0f32; 4];
    let chunks = q.len() / 4;
    for c in 0..chunks {
        let i = c * 4;
        acc[0] += q[i] * codes[i] as f32;
        acc[1] += q[i + 1] * codes[i + 1] as f32;
        acc[2] += q[i + 2] * codes[i + 2] as f32;
        acc[3] += q[i + 3] * codes[i + 3] as f32;
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    for i in chunks * 4..q.len() {
        s += q[i] * codes[i] as f32;
    }
    s
}

fn dot_q_q(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| *x as i32 * *y as i32).sum()
}

/// The compact HNSW graph. Internal point ids are dense `u32`s in insertion
/// order; mapping them to caller ids is the wrapper's job.
pub struct CompactHnsw {
    dim: usize,
    ef_construction: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
    levels: Vec<u8>,
    links0: Vec<u32>,
    /// Point -> `level * M` slots for layers `1..=level`.
    upper: HashMap<u32, Box<[u32]>>,
    entry: u32,
    max_level: u8,
    rng: u64,
}

impl std::fmt::Debug for CompactHnsw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactHnsw")
            .field("dim", &self.dim)
            .field("len", &self.len())
            .field("max_level", &self.max_level)
            .finish_non_exhaustive()
    }
}

impl CompactHnsw {
    pub fn new(dim: usize, ef_construction: usize) -> Self {
        Self {
            dim,
            ef_construction: ef_construction.max(M0),
            codes: Vec::new(),
            scales: Vec::new(),
            levels: Vec::new(),
            links0: Vec::new(),
            upper: HashMap::new(),
            entry: NONE,
            max_level: 0,
            rng: 0x2545_F491_4F6C_DD1D,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Bytes held by the index's arrays (capacity, not just length), plus an
    /// estimate for the upper-layer map. Excludes the allocator's own overhead.
    pub fn memory_bytes(&self) -> usize {
        let upper: usize = self.upper.values().map(|s| s.len() * 4 + 16).sum::<usize>()
            + self.upper.capacity() * (4 + 16 + 1);
        self.codes.capacity()
            + self.scales.capacity() * 4
            + self.levels.capacity()
            + self.links0.capacity() * 4
            + upper
    }

    /// Release spare capacity left by growth (call after a bulk load).
    pub fn shrink_to_fit(&mut self) {
        self.codes.shrink_to_fit();
        self.scales.shrink_to_fit();
        self.levels.shrink_to_fit();
        self.links0.shrink_to_fit();
        self.upper.shrink_to_fit();
    }

    /// Reserve room for `additional` more points.
    pub fn reserve(&mut self, additional: usize) {
        self.codes.reserve(additional * self.dim);
        self.scales.reserve(additional);
        self.levels.reserve(additional);
        self.links0.reserve(additional * M0);
    }

    fn code(&self, id: u32) -> &[i8] {
        let s = id as usize * self.dim;
        &self.codes[s..s + self.dim]
    }

    fn sim_f(&self, q: &[f32], id: u32) -> f32 {
        dot_f_q(q, self.code(id)) * self.scales[id as usize]
    }

    fn sim_qq(&self, a: u32, b: u32) -> f32 {
        dot_q_q(self.code(a), self.code(b)) as f32
            * self.scales[a as usize]
            * self.scales[b as usize]
    }

    fn links(&self, id: u32, level: u8) -> &[u32] {
        if level == 0 {
            let s = id as usize * M0;
            &self.links0[s..s + M0]
        } else {
            let s = (level as usize - 1) * M;
            &self.upper[&id][s..s + M]
        }
    }

    fn links_mut(&mut self, id: u32, level: u8) -> &mut [u32] {
        if level == 0 {
            let s = id as usize * M0;
            &mut self.links0[s..s + M0]
        } else {
            let s = (level as usize - 1) * M;
            &mut self.upper.get_mut(&id).expect("upper links")[s..s + M]
        }
    }

    fn random_level(&mut self) -> u8 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let r = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let u = ((r >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let ml = 1.0 / (M as f64).ln();
        ((-u.ln() * ml).floor() as u64).min(MAX_LEVEL as u64) as u8
    }

    /// Greedy walk on one layer towards `q`.
    fn greedy(&self, q: &[f32], mut cur: Cand, level: u8) -> Cand {
        loop {
            let mut moved = false;
            for &n in self.links(cur.id, level) {
                if n == NONE {
                    break;
                }
                let s = self.sim_f(q, n);
                if s > cur.sim {
                    cur = Cand { sim: s, id: n };
                    moved = true;
                }
            }
            if !moved {
                return cur;
            }
        }
    }

    /// Best-first search on one layer; returns up to `ef` points, best first.
    fn search_layer(&self, q: &[f32], entry: Cand, ef: usize, level: u8) -> Vec<Cand> {
        let mut visited = Visited::new();
        visited.insert(entry.id);
        let mut frontier = BinaryHeap::new(); // best on top
        let mut found: BinaryHeap<Worst> = BinaryHeap::new(); // worst on top
        frontier.push(entry);
        found.push(Worst(std::cmp::Reverse(entry)));
        while let Some(c) = frontier.pop() {
            let worst = found.peek().map(|w| w.0 .0.sim).unwrap_or(f32::MIN);
            if c.sim < worst && found.len() >= ef {
                break;
            }
            for &n in self.links(c.id, level) {
                if n == NONE {
                    break;
                }
                if !visited.insert(n) {
                    continue;
                }
                let s = self.sim_f(q, n);
                let worst = found.peek().map(|w| w.0 .0.sim).unwrap_or(f32::MIN);
                if found.len() < ef || s > worst {
                    let cand = Cand { sim: s, id: n };
                    frontier.push(cand);
                    found.push(Worst(std::cmp::Reverse(cand)));
                    if found.len() > ef {
                        found.pop();
                    }
                }
            }
        }
        let mut out: Vec<Cand> = found.into_iter().map(|w| w.0 .0).collect();
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }

    /// Neighbour selection heuristic over `cands` (best first, similarities
    /// to the base point): keep a candidate only when it is closer to the
    /// base than to every neighbour already kept; then fill free slots with
    /// the nearest pruned candidates.
    fn select(&self, cands: &[Cand], max: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = Vec::with_capacity(max);
        let mut pruned: Vec<u32> = Vec::new();
        for c in cands {
            if kept.len() >= max {
                break;
            }
            if kept.iter().all(|&r| self.sim_qq(c.id, r) < c.sim) {
                kept.push(c.id);
            } else {
                pruned.push(c.id);
            }
        }
        for p in pruned {
            if kept.len() >= max {
                break;
            }
            kept.push(p);
        }
        kept
    }

    /// Add the link `from -> to` on `level`, re-selecting `from`'s neighbours
    /// when its slots are full.
    fn add_link(&mut self, from: u32, to: u32, level: u8) {
        let max = if level == 0 { M0 } else { M };
        let slots = self.links(from, level);
        if let Some(free) = slots.iter().position(|&n| n == NONE) {
            self.links_mut(from, level)[free] = to;
            return;
        }
        let mut cands: Vec<Cand> = slots
            .iter()
            .chain(std::iter::once(&to))
            .map(|&n| Cand {
                sim: self.sim_qq(from, n),
                id: n,
            })
            .collect();
        cands.sort_unstable_by(|a, b| b.cmp(a));
        let chosen = self.select(&cands, max);
        let out = self.links_mut(from, level);
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = chosen.get(i).copied().unwrap_or(NONE);
        }
    }

    /// Insert a vector; returns its internal id.
    pub fn insert(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim, "vector dimension");
        let id = self.levels.len() as u32;
        assert!(id != NONE, "compact index is full");
        let (unit, codes, scale) = quantise(v);
        let level = self.random_level();
        self.codes.extend_from_slice(&codes);
        self.scales.push(scale);
        self.levels.push(level);
        self.links0.extend(std::iter::repeat(NONE).take(M0));
        if level > 0 {
            self.upper
                .insert(id, vec![NONE; level as usize * M].into_boxed_slice());
        }
        if self.entry == NONE {
            self.entry = id;
            self.max_level = level;
            return id;
        }
        let mut ep = Cand {
            sim: self.sim_f(&unit, self.entry),
            id: self.entry,
        };
        let mut l = self.max_level;
        while l > level {
            ep = self.greedy(&unit, ep, l);
            l -= 1;
        }
        let top = level.min(self.max_level);
        for l in (0..=top).rev() {
            let found = self.search_layer(&unit, ep, self.ef_construction, l);
            let max = if l == 0 { M0 } else { M };
            let chosen = self.select(&found, max);
            {
                let slots = self.links_mut(id, l);
                for (i, n) in chosen.iter().enumerate() {
                    slots[i] = *n;
                }
            }
            for &n in &chosen {
                self.add_link(n, id, l);
            }
            if let Some(best) = found.first() {
                ep = *best;
            }
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry = id;
        }
        id
    }

    /// The `k` nearest points to `q` by cosine similarity, best first, as
    /// (internal id, similarity). `ef` is the search breadth (raised to `k`).
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.entry == NONE || k == 0 || q.len() != self.dim {
            return Vec::new();
        }
        let (unit, _, _) = quantise(q);
        let mut ep = Cand {
            sim: self.sim_f(&unit, self.entry),
            id: self.entry,
        };
        let mut l = self.max_level;
        while l > 0 {
            ep = self.greedy(&unit, ep, l);
            l -= 1;
        }
        let found = self.search_layer(&unit, ep, ef.max(k), 0);
        found.into_iter().take(k).map(|c| (c.id, c.sim)).collect()
    }

    /// Exact search over every point (for tests and quality checks).
    pub fn brute_force(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        let (unit, _, _) = quantise(q);
        let mut all: Vec<Cand> = (0..self.len() as u32)
            .map(|id| Cand {
                sim: self.sim_f(&unit, id),
                id,
            })
            .collect();
        all.sort_unstable_by(|a, b| b.cmp(a));
        all.into_iter().take(k).map(|c| (c.id, c.sim)).collect()
    }

    /// Write the index to `w` (little-endian, versioned).
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        for v in [
            1u32,
            self.dim as u32,
            M as u32,
            M0 as u32,
            self.ef_construction as u32,
            self.entry,
        ] {
            w.write_all(&v.to_le_bytes())?;
        }
        w.write_all(&[self.max_level])?;
        w.write_all(&(self.len() as u64).to_le_bytes())?;
        w.write_all(&self.rng.to_le_bytes())?;
        for s in &self.scales {
            w.write_all(&s.to_le_bytes())?;
        }
        w.write_all(&self.levels)?;
        // i8 -> u8 is a bit-for-bit reinterpretation.
        let bytes: Vec<u8> = self.codes.iter().map(|c| *c as u8).collect();
        w.write_all(&bytes)?;
        let mut buf = Vec::with_capacity(M0 * 4 * 4096);
        for chunk in self.links0.chunks(M0 * 4096) {
            buf.clear();
            for l in chunk {
                buf.extend_from_slice(&l.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
        let mut upper: Vec<(&u32, &Box<[u32]>)> = self.upper.iter().collect();
        upper.sort_unstable_by_key(|(id, _)| **id);
        w.write_all(&(upper.len() as u64).to_le_bytes())?;
        for (id, slots) in upper {
            w.write_all(&id.to_le_bytes())?;
            w.write_all(&(slots.len() as u32).to_le_bytes())?;
            for s in slots.iter() {
                w.write_all(&s.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Read an index written by [`CompactHnsw::write_to`].
    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        fn bad(msg: &str) -> io::Error {
            io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
        }
        fn u32_(r: &mut impl Read) -> io::Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        }
        fn u64_(r: &mut impl Read) -> io::Result<u64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        }
        fn u32s(r: &mut impl Read, n: usize) -> io::Result<Vec<u32>> {
            let mut bytes = vec![0u8; n * 4];
            r.read_exact(&mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not a compact hnsw file"));
        }
        let version = u32_(r)?;
        if version != 1 {
            return Err(bad("unknown compact hnsw version"));
        }
        let dim = u32_(r)? as usize;
        let (m, m0) = (u32_(r)? as usize, u32_(r)? as usize);
        if m != M || m0 != M0 {
            return Err(bad("compact hnsw degree differs from this build"));
        }
        let ef_construction = u32_(r)? as usize;
        let entry = u32_(r)?;
        let mut lvl = [0u8; 1];
        r.read_exact(&mut lvl)?;
        let n = u64_(r)? as usize;
        let rng = u64_(r)?;
        let scales: Vec<f32> = u32s(r, n)?.into_iter().map(f32::from_bits).collect();
        let mut levels = vec![0u8; n];
        r.read_exact(&mut levels)?;
        let mut raw = vec![0u8; n * dim];
        r.read_exact(&mut raw)?;
        let codes: Vec<i8> = raw.into_iter().map(|b| b as i8).collect();
        let links0 = u32s(r, n * M0)?;
        let n_upper = u64_(r)? as usize;
        let mut upper = HashMap::with_capacity(n_upper);
        for _ in 0..n_upper {
            let id = u32_(r)?;
            let len = u32_(r)? as usize;
            if (id as usize) >= n || len != levels[id as usize] as usize * M {
                return Err(bad("compact hnsw upper links do not match levels"));
            }
            upper.insert(id, u32s(r, len)?.into_boxed_slice());
        }
        Ok(Self {
            dim,
            ef_construction,
            codes,
            scales,
            levels,
            links0,
            upper,
            entry,
            max_level: lvl[0],
            rng,
        })
    }

    /// Save to `path` atomically (temporary file, then rename).
    pub fn save(
        &self,
        path: &Path,
        extra: impl FnOnce(&mut BufWriter<std::fs::File>) -> io::Result<()>,
    ) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
            self.write_to(&mut w)?;
            extra(&mut w)?;
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Load from `path`; `extra` reads whatever [`CompactHnsw::save`]'s
    /// `extra` wrote after the index.
    pub fn load<T>(
        path: &Path,
        extra: impl FnOnce(&mut BufReader<std::fs::File>) -> io::Result<T>,
    ) -> io::Result<(Self, T)> {
        let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
        let idx = Self::read_from(&mut r)?;
        let t = extra(&mut r)?;
        Ok((idx, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| (0..dim).map(|_| lcg(&mut s)).collect())
            .collect()
    }

    #[test]
    fn finds_itself_and_orders_by_similarity() {
        let mut idx = CompactHnsw::new(3, 64);
        idx.insert(&[1.0, 0.0, 0.0]);
        idx.insert(&[0.95, 0.1, 0.0]);
        idx.insert(&[0.0, 1.0, 0.0]);
        let hits = idx.search(&[1.0, 0.0, 0.0], 3, 16);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 1);
        assert_eq!(hits[2].0, 2);
        assert!((hits[0].1 - 1.0).abs() < 0.02);
    }

    #[test]
    fn empty_and_zero_vectors_are_safe() {
        let idx = CompactHnsw::new(4, 64);
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5, 16).is_empty());
        let mut idx = CompactHnsw::new(4, 64);
        idx.insert(&[0.0; 4]);
        idx.insert(&[1.0, 0.0, 0.0, 0.0]);
        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2, 16);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn recall_against_brute_force() {
        let dim = 32;
        let data = random_vectors(3000, dim, 7);
        let mut idx = CompactHnsw::new(dim, 200);
        for v in &data {
            idx.insert(v);
        }
        let queries = random_vectors(50, dim, 99);
        let mut hit = 0usize;
        for q in &queries {
            let truth: Vec<u32> = idx.brute_force(q, 10).into_iter().map(|h| h.0).collect();
            let got: Vec<u32> = idx.search(q, 10, 64).into_iter().map(|h| h.0).collect();
            hit += truth.iter().filter(|t| got.contains(t)).count();
        }
        let recall = hit as f32 / (queries.len() * 10) as f32;
        assert!(recall > 0.95, "recall@10 {recall}");
    }

    #[test]
    fn save_and_load_round_trip() {
        let dim = 16;
        let data = random_vectors(500, dim, 3);
        let mut idx = CompactHnsw::new(dim, 100);
        for v in &data {
            idx.insert(v);
        }
        let mut bytes = Vec::new();
        idx.write_to(&mut bytes).unwrap();
        let back = CompactHnsw::read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(back.len(), idx.len());
        let q = &data[42];
        assert_eq!(idx.search(q, 10, 64), back.search(q, 10, 64));
        assert_eq!(back.search(q, 1, 64)[0].0, 42);
        // and the loaded index keeps growing
        let mut back = back;
        let id = back.insert(&data[0]);
        assert_eq!(id as usize, data.len());
    }

    #[test]
    fn rejects_garbage() {
        assert!(CompactHnsw::read_from(&mut &b"nonsense-bytes-here"[..]).is_err());
    }

    #[test]
    fn memory_is_a_few_hundred_bytes_per_128d_point() {
        let dim = 128;
        let data = random_vectors(2000, dim, 11);
        let mut idx = CompactHnsw::new(dim, 100);
        idx.reserve(data.len());
        for v in &data {
            idx.insert(v);
        }
        idx.shrink_to_fit();
        let per = idx.memory_bytes() / idx.len();
        assert!(per < 300, "{per} bytes per point");
    }
}
