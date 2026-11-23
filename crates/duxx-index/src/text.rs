//! Tantivy-backed BM25 text index with batched commits.
//!
//! Phase 2.5: instead of committing tantivy on every `insert` (slow:
//! ~4 ms each), we commit every `commit_every` inserts. Reads (`search`,
//! `len`) auto-flush any pending writes so the "insert-then-search"
//! contract still holds.
//!
//! Defaults to `commit_every = 100`. Override at construction with
//! [`TextIndex::with_commit_every`].
//!
//! ## Fields
//!
//! Every document is indexed in three fields: `text` (the whole text, exactly
//! as in 0.4.3 and earlier), `title` and `url`. A query is scored as BM25 over
//! `text`, plus `title` and `url` matches weighted by [`FieldBoosts`], plus a
//! bonus when the whole query appears as a phrase in the title. A document
//! with no title and no URL matches only in `text`, so it scores exactly as
//! before. The title and URL come from the caller ([`DocFields`]) or, when
//! the caller gives none, from the text's own layout: a text whose first line
//! is an `http(s)://` URL has that URL as its `url` and its second line as
//! its `title` (see [`DocFields::from_layout`]).

use duxx_core::{Error, Result};
use parking_lot::Mutex;
use std::path::Path;
use std::str::CharIndices;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tantivy::{
    collector::{Collector, TopDocs},
    directory::MmapDirectory,
    doc,
    query::{
        BooleanQuery, BoostQuery, EnableScoring, Occur, PhraseQuery, Query, QueryParser, TermQuery,
    },
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, FAST, INDEXED,
        STORED, TEXT,
    },
    tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenStream, Tokenizer},
    DocAddress, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, TantivyDocument,
    TantivyError, Term,
};

/// Default insert count between forced commits.
pub const DEFAULT_COMMIT_EVERY: usize = 100;

/// Default weight of a title match (`DUXX_TEXT_TITLE_BOOST`).
pub const DEFAULT_TITLE_BOOST: f32 = 3.0;
/// Default weight of a URL match (`DUXX_TEXT_URL_BOOST`).
pub const DEFAULT_URL_BOOST: f32 = 1.5;
/// Default weight of the whole query appearing as a phrase in the title
/// (`DUXX_TEXT_TITLE_PHRASE_BOOST`).
pub const DEFAULT_TITLE_PHRASE_BOOST: f32 = 2.0;

/// How much a match in each field counts, relative to a match in the body
/// text (which always counts 1). A boost of 0 leaves that field out of the
/// query entirely; [`FieldBoosts::OFF`] scores every document exactly as
/// 0.4.3 did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldBoosts {
    /// Weight of a match in the title.
    pub title: f32,
    /// Weight of a match in the URL.
    pub url: f32,
    /// Weight of the whole query (two words or more) appearing, in order and
    /// adjacent, in the title. Added on top of the per-word title matches.
    pub title_phrase: f32,
}

impl FieldBoosts {
    /// Body text only: the scoring of 0.4.3 and earlier.
    pub const OFF: FieldBoosts = FieldBoosts {
        title: 0.0,
        url: 0.0,
        title_phrase: 0.0,
    };

    /// `DUXX_TEXT_TITLE_BOOST`, `DUXX_TEXT_URL_BOOST` and
    /// `DUXX_TEXT_TITLE_PHRASE_BOOST`, each a number >= 0. Unset (or not a
    /// number >= 0, which is logged) means the default: 3, 1.5 and 2.
    pub fn from_env() -> Self {
        FieldBoosts {
            title: env_weight("DUXX_TEXT_TITLE_BOOST", DEFAULT_TITLE_BOOST),
            url: env_weight("DUXX_TEXT_URL_BOOST", DEFAULT_URL_BOOST),
            title_phrase: env_weight("DUXX_TEXT_TITLE_PHRASE_BOOST", DEFAULT_TITLE_PHRASE_BOOST),
        }
    }

    /// Whether no field but the body text is scored.
    pub fn is_off(&self) -> bool {
        self.title <= 0.0 && self.url <= 0.0 && self.title_phrase <= 0.0
    }
}

impl Default for FieldBoosts {
    fn default() -> Self {
        FieldBoosts {
            title: DEFAULT_TITLE_BOOST,
            url: DEFAULT_URL_BOOST,
            title_phrase: DEFAULT_TITLE_PHRASE_BOOST,
        }
    }
}

/// A non-negative number from the environment variable `name`, or `default`
/// when it is unset or invalid (logged).
pub fn env_weight(name: &str, default: f32) -> f32 {
    match std::env::var(name) {
        Ok(v) => match v.trim().parse::<f32>() {
            Ok(x) if x.is_finite() && x >= 0.0 => x,
            _ => {
                tracing::warn!("{name}={v:?} is not a number >= 0; using the default {default}");
                default
            }
        },
        Err(_) => default,
    }
}

/// The title and URL of a document, indexed beside its body text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DocFields {
    pub title: Option<String>,
    pub url: Option<String>,
}

impl DocFields {
    /// The fields a text's own layout gives: when its first line is an
    /// `http://` or `https://` URL (one word, no spaces), that line is the
    /// URL and the second line, if any, the title. Any other text has no
    /// fields. This is the layout Gaelxe writes: URL, title, description,
    /// body, one per line.
    pub fn from_layout(text: &str) -> Self {
        let mut lines = text.lines();
        let first = lines.next().unwrap_or("").trim();
        let lower = first.get(..8).map(str::to_ascii_lowercase);
        let is_url = matches!(lower.as_deref(), Some(p) if p.starts_with("http://") || p == "https://")
            && first.len() > 8
            && !first.chars().any(char::is_whitespace);
        if !is_url {
            return DocFields::default();
        }
        let title = lines
            .next()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string);
        DocFields {
            title,
            url: Some(first.to_string()),
        }
    }

    /// These fields where given, the text's layout ([`DocFields::from_layout`])
    /// for any field not given.
    pub fn or_layout(&self, text: &str) -> Self {
        if self.title.is_some() && self.url.is_some() {
            return self.clone();
        }
        let layout = DocFields::from_layout(text);
        DocFields {
            title: self.title.clone().or(layout.title),
            url: self.url.clone().or(layout.url),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.url.is_none()
    }
}

/// The part of a URL worth matching: without the scheme and a leading
/// `www.`. The analyzer then splits it at dots, slashes and hyphens, so
/// `https://www.digilocker.gov.in/` is indexed as `digilocker gov in`.
fn url_words(url: &str) -> &str {
    let rest = match url.find("://") {
        Some(i) => &url[i + 3..],
        None => url,
    };
    match rest.get(..4) {
        Some(p) if p.eq_ignore_ascii_case("www.") => &rest[4..],
        _ => rest,
    }
}

/// Name under which the [`TextTokenizer::Words`] analyzer is registered.
pub const WORDS_TOKENIZER: &str = "duxx_words";

/// How the text field splits text into terms. Chosen with the
/// `DUXX_TEXT_TOKENIZER` environment variable (see [`TextTokenizer::from_env`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextTokenizer {
    /// tantivy's default analyzer, as in 0.4.2 and earlier: a word ends at
    /// every character that is not alphanumeric, words of 40 bytes or more
    /// are dropped, and words are lower-cased.
    Default,
    /// Words keep their combining marks and joiners. Indian scripts write
    /// a virama (्), a nukta (़) and zero-width joiners inside words; the
    /// default analyzer splits there, so "शिक्षा" is indexed as "शिक" and
    /// "षा" and a query for either fragment matches it. Words up to 100
    /// bytes are kept (an Indian-script letter takes three bytes in UTF-8,
    /// so the default 40-byte limit dropped long Malayalam or Tamil words).
    /// Latin text splits exactly as before.
    Words,
}

impl TextTokenizer {
    /// `DUXX_TEXT_TOKENIZER=words` selects [`TextTokenizer::Words`]; unset or
    /// anything else keeps [`TextTokenizer::Default`].
    pub fn from_env() -> Self {
        match std::env::var("DUXX_TEXT_TOKENIZER") {
            Ok(v) if v.trim().eq_ignore_ascii_case("words") => TextTokenizer::Words,
            _ => TextTokenizer::Default,
        }
    }

    /// The tokenizer name stored in the index schema.
    pub fn name(self) -> &'static str {
        match self {
            TextTokenizer::Default => "default",
            TextTokenizer::Words => WORDS_TOKENIZER,
        }
    }
}

/// Whether `c` belongs inside a word for [`TextTokenizer::Words`]: letters
/// and digits, every combining character of the Indian scripts (U+0900 to
/// U+0DFF, apart from the danda, the double danda and the abbreviation
/// sign, which end sentences and words), and the zero-width non-joiner and
/// joiner (U+200C, U+200D).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
        || (('\u{0900}'..='\u{0DFF}').contains(&c)
            && !matches!(c, '\u{0964}' | '\u{0965}' | '\u{0970}'))
        || c == '\u{200C}'
        || c == '\u{200D}'
}

/// Splits text into words made of [`is_word_char`] characters.
#[derive(Clone, Default)]
pub struct WordsTokenizer {
    token: Token,
}

/// Token stream of [`WordsTokenizer`].
pub struct WordsTokenStream<'a> {
    text: &'a str,
    chars: CharIndices<'a>,
    token: &'a mut Token,
}

impl Tokenizer for WordsTokenizer {
    type TokenStream<'a> = WordsTokenStream<'a>;
    fn token_stream<'a>(&'a mut self, text: &'a str) -> WordsTokenStream<'a> {
        self.token.reset();
        WordsTokenStream {
            text,
            chars: text.char_indices(),
            token: &mut self.token,
        }
    }
}

impl WordsTokenStream<'_> {
    fn search_token_end(&mut self) -> usize {
        self.chars
            .find(|(_, c)| !is_word_char(*c))
            .map(|(offset, _)| offset)
            .unwrap_or(self.text.len())
    }
}

impl TokenStream for WordsTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        self.token.position = self.token.position.wrapping_add(1);
        while let Some((offset_from, c)) = self.chars.next() {
            if is_word_char(c) {
                let offset_to = self.search_token_end();
                self.token.offset_from = offset_from;
                self.token.offset_to = offset_to;
                self.token.text.push_str(&self.text[offset_from..offset_to]);
                return true;
            }
        }
        false
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

/// The analyzer registered as [`WORDS_TOKENIZER`].
pub fn words_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(WordsTokenizer::default())
        .filter(RemoveLongFilter::limit(100))
        .filter(LowerCaser)
        .build()
}

/// The fields of the index schema.
#[derive(Clone, Copy, Debug)]
struct Fields {
    id: Field,
    text: Field,
    title: Field,
    url: Field,
}

/// BM25 full-text index, in-memory, Send + Sync.
pub struct TextIndex {
    id_field: Field,
    text_field: Field,
    title_field: Field,
    url_field: Field,
    boosts: FieldBoosts,
    index: Index,
    writer: Arc<Mutex<IndexWriter>>,
    reader: IndexReader,
    pending: AtomicUsize,
    commit_every: usize,
}

impl TextIndex {
    /// Build with the default commit threshold ([`DEFAULT_COMMIT_EVERY`]).
    pub fn new() -> Self {
        Self::with_commit_every(DEFAULT_COMMIT_EVERY)
    }

    /// Build with a custom commit threshold. Use a low value (1) when
    /// every insert must be immediately durable; higher values amortize
    /// commit overhead across batches.
    pub fn with_commit_every(commit_every: usize) -> Self {
        Self::try_with(commit_every).expect("tantivy in-RAM index init must not fail")
    }

    fn try_with(commit_every: usize) -> Result<Self> {
        Self::in_ram_with(commit_every, TextTokenizer::from_env())
    }

    /// In-RAM index with an explicit tokenizer instead of the one
    /// `DUXX_TEXT_TOKENIZER` names.
    pub fn in_ram_with(commit_every: usize, tokenizer: TextTokenizer) -> Result<Self> {
        let (fields, schema) = build_schema(tokenizer);
        let index = Index::create_in_ram(schema);
        Self::from_index(index, fields, commit_every)
    }

    /// Open a disk-backed tantivy index at `dir`. The directory is
    /// created if missing. Reopening an existing dir restores the
    /// previously written state — the writer queue is empty, so the
    /// caller can immediately commit new docs without rebuild.
    ///
    /// The tokenizer comes from `DUXX_TEXT_TOKENIZER`. When the index on
    /// disk was built with another tokenizer, or with another schema (an
    /// index written by 0.4.3 or earlier has no `title` and `url` fields), it
    /// is recreated empty: the text index is derived from the rows, and
    /// `MemoryStore::open_at` refills it from the row store because its
    /// count no longer matches.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_commit_every(dir, DEFAULT_COMMIT_EVERY)
    }

    pub fn open_with_commit_every(dir: impl AsRef<Path>, commit_every: usize) -> Result<Self> {
        Self::open_with(dir, commit_every, TextTokenizer::from_env())
    }

    /// [`TextIndex::open_with_commit_every`] with an explicit tokenizer.
    pub fn open_with(
        dir: impl AsRef<Path>,
        commit_every: usize,
        tokenizer: TextTokenizer,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::Index(format!("create tantivy dir: {e}")))?;
        let (fields, schema) = build_schema(tokenizer);
        let mmap =
            MmapDirectory::open(dir).map_err(|e| Error::Index(format!("tantivy mmap dir: {e}")))?;
        let index = match Index::open_or_create(mmap, schema.clone()) {
            Ok(index) => index,
            Err(TantivyError::SchemaError(why)) => {
                tracing::warn!(
                    dir = %dir.display(),
                    tokenizer = tokenizer.name(),
                    "text index schema differs ({why}); recreating it empty to be refilled"
                );
                std::fs::remove_dir_all(dir)
                    .map_err(|e| Error::Index(format!("remove old tantivy dir: {e}")))?;
                std::fs::create_dir_all(dir)
                    .map_err(|e| Error::Index(format!("create tantivy dir: {e}")))?;
                let mmap = MmapDirectory::open(dir)
                    .map_err(|e| Error::Index(format!("tantivy mmap dir: {e}")))?;
                Index::create(mmap, schema, tantivy::IndexSettings::default())
                    .map_err(|e| Error::Index(format!("tantivy create: {e}")))?
            }
            Err(e) => return Err(Error::Index(format!("tantivy open_or_create: {e}"))),
        };
        Self::from_index(index, fields, commit_every)
    }

    /// The tokenizer the text field was built with.
    pub fn tokenizer(&self) -> TextTokenizer {
        let name = match self
            .index
            .schema()
            .get_field_entry(self.text_field)
            .field_type()
        {
            tantivy::schema::FieldType::Str(opts) => opts
                .get_indexing_options()
                .map(|i| i.tokenizer().to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        if name == WORDS_TOKENIZER {
            TextTokenizer::Words
        } else {
            TextTokenizer::Default
        }
    }

    fn from_index(index: Index, fields: Fields, commit_every: usize) -> Result<Self> {
        // Registered on every index, so one built with either tokenizer opens.
        index
            .tokenizers()
            .register(WORDS_TOKENIZER, words_analyzer());
        let writer: IndexWriter = index
            .writer(50_000_000)
            .map_err(|e| Error::Index(format!("tantivy writer: {e}")))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::Index(format!("tantivy reader: {e}")))?;

        Ok(Self {
            id_field: fields.id,
            text_field: fields.text,
            title_field: fields.title,
            url_field: fields.url,
            boosts: FieldBoosts::from_env(),
            index,
            writer: Arc::new(Mutex::new(writer)),
            reader,
            pending: AtomicUsize::new(0),
            commit_every: commit_every.max(1),
        })
    }

    /// How title, URL and title-phrase matches are weighted (see
    /// [`FieldBoosts`]). Read from the environment at construction.
    pub fn field_boosts(&self) -> FieldBoosts {
        self.boosts
    }

    /// Change how title, URL and title-phrase matches are weighted. Takes
    /// effect on the next search; the index itself does not change.
    pub fn set_field_boosts(&mut self, boosts: FieldBoosts) {
        self.boosts = boosts;
    }

    /// The tantivy document for one row: the whole text in `text`, and the
    /// title and URL (given, else from the text's layout) in their fields.
    fn make_doc(&self, id: u64, text: String, fields: &DocFields) -> TantivyDocument {
        let fields = fields.or_layout(&text);
        let mut d = doc!(self.id_field => id, self.text_field => text);
        if let Some(title) = fields.title {
            d.add_text(self.title_field, title);
        }
        if let Some(url) = fields.url {
            d.add_text(self.url_field, url_words(&url));
        }
        d
    }

    /// Insert a document. Commits if the threshold is reached. Its title and
    /// URL come from its layout ([`DocFields::from_layout`]).
    pub fn insert(&mut self, id: u64, text: String) -> Result<()> {
        self.insert_with(id, text, &DocFields::default())
    }

    /// Insert a document with a title and URL given by the caller. A field
    /// left `None` still comes from the text's layout.
    pub fn insert_with(&mut self, id: u64, text: String, fields: &DocFields) -> Result<()> {
        let d = self.make_doc(id, text, fields);
        let mut w = self.writer.lock();
        w.add_document(d)
            .map_err(|e| Error::Index(format!("add_document: {e}")))?;
        let n = self.pending.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= self.commit_every {
            w.commit()
                .map_err(|e| Error::Index(format!("commit: {e}")))?;
            self.pending.store(0, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Rebuild the index over exactly `survivors`, dropping every other
    /// document. Counterpart to `VectorIndex::rebuild` — called by
    /// `MemoryStore::compact` so the BM25 postings stop referencing
    /// forgotten rows. Works on both in-RAM and disk-backed indices
    /// (it deletes all docs and re-adds the survivors in one commit, so a
    /// disk-backed directory is reused rather than orphaned).
    pub fn rebuild(&mut self, survivors: &[(u64, String)]) -> Result<()> {
        let docs: Vec<TantivyDocument> = survivors
            .iter()
            .map(|(id, text)| self.make_doc(*id, text.clone(), &DocFields::default()))
            .collect();
        self.rebuild_from(docs)
    }

    /// [`TextIndex::rebuild`] for documents with a title and URL given by
    /// the caller.
    pub fn rebuild_with(&mut self, survivors: &[(u64, String, DocFields)]) -> Result<()> {
        let docs: Vec<TantivyDocument> = survivors
            .iter()
            .map(|(id, text, fields)| self.make_doc(*id, text.clone(), fields))
            .collect();
        self.rebuild_from(docs)
    }

    fn rebuild_from(&mut self, docs: Vec<TantivyDocument>) -> Result<()> {
        let mut w = self.writer.lock();
        // Flush any batched-but-uncommitted inserts into segments first, so
        // `delete_all_documents` (which only targets committed docs) actually
        // removes them. Without this, pending docs survive the rebuild.
        w.commit()
            .map_err(|e| Error::Index(format!("commit (flush pending): {e}")))?;
        w.delete_all_documents()
            .map_err(|e| Error::Index(format!("delete_all_documents: {e}")))?;
        for d in docs {
            w.add_document(d)
                .map_err(|e| Error::Index(format!("add_document: {e}")))?;
        }
        // Single commit applies the delete (to the pre-delete docs) and the
        // survivor adds (higher opstamp, so they're kept).
        w.commit()
            .map_err(|e| Error::Index(format!("commit: {e}")))?;
        drop(w);
        self.pending.store(0, Ordering::SeqCst);
        self.reader
            .reload()
            .map_err(|e| Error::Index(format!("reader reload: {e}")))?;
        Ok(())
    }

    /// Change how many inserts may wait between commits. A bulk load sets a
    /// large value (each commit writes a segment and syncs it), then puts the
    /// normal value back.
    pub fn set_commit_every(&mut self, commit_every: usize) {
        self.commit_every = commit_every.max(1);
    }

    /// Force-commit any pending writes. Idempotent.
    pub fn flush(&self) -> Result<()> {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return Ok(());
        }
        let mut w = self.writer.lock();
        w.commit()
            .map_err(|e| Error::Index(format!("commit: {e}")))?;
        self.pending.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// A searcher over everything inserted so far: pending writes are
    /// committed and the reader reloaded first.
    fn fresh_searcher(&self) -> Option<Searcher> {
        if let Err(e) = self.flush() {
            tracing::warn!("flush before search failed: {e}");
            return None;
        }
        if let Err(e) = self.reader.reload() {
            tracing::warn!("tantivy reader reload failed: {e}");
            return None;
        }
        Some(self.reader.searcher())
    }

    /// BM25 search; auto-flushes pending writes first. The body text counts
    /// 1, title and URL matches count as [`FieldBoosts`] says, and the whole
    /// query appearing as a phrase in the title adds a bonus.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        self.run(query, k, true)
    }

    /// How well the query matches each document's title and URL alone, with
    /// the title-phrase bonus, best first: the part of [`TextIndex::search`]'s
    /// score that does not come from the body text. Empty when every field
    /// boost is 0 or no document has a title or URL that matches.
    pub fn field_search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        self.run(query, k, false)
    }

    fn run(&self, query: &str, k: usize, with_body: bool) -> Vec<(u64, f32)> {
        let Some(searcher) = self.fresh_searcher() else {
            return Vec::new();
        };
        let Some(q) = self.scored_query(query, with_body) else {
            return Vec::new();
        };
        match searcher.search(&q, &TopDocs::with_limit(k)) {
            Ok(docs) => self.ids(&searcher, docs),
            Err(e) => {
                tracing::warn!("tantivy search failed: {e}");
                Vec::new()
            }
        }
    }

    /// [`TextIndex::search`] that stops at `deadline`. The index's segments
    /// are searched one at a time; the best `k` hits of each finished segment
    /// are appended to `found` (unsorted, at most `k` per segment), so a
    /// caller that stops waiting at the deadline can use what is there.
    /// Returns `true` when every segment was searched, `false` when the
    /// deadline came first.
    pub fn search_until(
        &self,
        query: &str,
        k: usize,
        deadline: Instant,
        found: &Mutex<Vec<(u64, f32)>>,
    ) -> bool {
        let Some(searcher) = self.fresh_searcher() else {
            return true;
        };
        let Some(q) = self.scored_query(query, true) else {
            return true;
        };
        let weight = match q.weight(EnableScoring::enabled_from_searcher(&searcher)) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("tantivy weight failed: {e}");
                return true;
            }
        };
        let collector = TopDocs::with_limit(k);
        for (ord, segment) in searcher.segment_readers().iter().enumerate() {
            if Instant::now() >= deadline {
                return false;
            }
            match collector.collect_segment(weight.as_ref(), ord as u32, segment) {
                Ok(docs) => {
                    let hits = self.ids(&searcher, docs);
                    found.lock().extend(hits);
                }
                Err(e) => tracing::warn!("tantivy segment search failed: {e}"),
            }
        }
        true
    }

    /// Row ids and scores of tantivy hits.
    fn ids(&self, searcher: &Searcher, docs: Vec<(f32, DocAddress)>) -> Vec<(u64, f32)> {
        docs.into_iter()
            .filter_map(|(score, addr)| {
                let d: TantivyDocument = searcher.doc(addr).ok()?;
                let v = d.get_first(self.id_field)?;
                let id = v.as_u64()?;
                Some((id, score))
            })
            .collect()
    }

    /// The query to run: the words (or query syntax) over the body text when
    /// `with_body`, the title and the URL, each with its boost, plus the
    /// title-phrase bonus. With [`FieldBoosts::OFF`] and `with_body` this is
    /// exactly the body-only query of 0.4.3.
    fn scored_query(&self, query: &str, with_body: bool) -> Option<Box<dyn Query>> {
        let b = self.boosts;
        let mut fields: Vec<(Field, f32)> = Vec::with_capacity(3);
        if with_body {
            fields.push((self.text_field, 1.0));
        }
        if b.title > 0.0 {
            fields.push((self.title_field, b.title));
        }
        if b.url > 0.0 {
            fields.push((self.url_field, b.url));
        }
        let words = if fields.is_empty() {
            None
        } else {
            self.parsed_query(query, &fields)
        };
        let phrase: Option<Box<dyn Query>> = if b.title_phrase > 0.0 {
            self.title_phrase_query(query)
                .map(|p| Box::new(BoostQuery::new(p, b.title_phrase)) as Box<dyn Query>)
        } else {
            None
        };
        match (words, phrase) {
            (Some(w), None) => Some(w),
            (None, p) => p,
            (Some(w), Some(p)) => {
                // With the body, the phrase only adds to documents the query
                // matches (so `-word` exclusions still hold).
                let occur = if with_body {
                    Occur::Must
                } else {
                    Occur::Should
                };
                Some(Box::new(BooleanQuery::new(vec![
                    (occur, w),
                    (Occur::Should, p),
                ])))
            }
        }
    }

    /// The query parsed as tantivy query syntax over `fields`, each with its
    /// boost. Plain text often is not valid query syntax: a title with a
    /// colon ("arrested: …" reads as a field name) or an unbalanced bracket
    /// or quote failed to parse, and the text side of recall returned
    /// nothing. Such a query is searched as its words instead.
    fn parsed_query(&self, query: &str, fields: &[(Field, f32)]) -> Option<Box<dyn Query>> {
        let mut parser =
            QueryParser::for_index(&self.index, fields.iter().map(|(f, _)| *f).collect());
        for (field, boost) in fields {
            if *boost != 1.0 {
                parser.set_field_boost(*field, *boost);
            }
        }
        match parser.parse_query(query) {
            Ok(q) => Some(q),
            Err(e) => {
                tracing::debug!("query parse failed ({e}); searching its words");
                self.words_query(query, fields)
            }
        }
    }

    /// The query's words, split by the text field's analyzer (every field
    /// uses the same one).
    fn query_words(&self, query: &str) -> Vec<String> {
        let Ok(mut analyzer) = self.index.tokenizer_for_field(self.text_field) else {
            return Vec::new();
        };
        let mut stream = analyzer.token_stream(query);
        let mut words = Vec::new();
        while stream.advance() {
            words.push(stream.token().text.clone());
        }
        words
    }

    /// Any of the query's words in any of `fields`, scored by BM25 with each
    /// field's boost: what the query parser makes of plain words, without
    /// its syntax. `None` when the query has no words.
    fn words_query(&self, query: &str, fields: &[(Field, f32)]) -> Option<Box<dyn Query>> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for word in self.query_words(query) {
            for (field, boost) in fields {
                let term = Term::from_field_text(*field, &word);
                let q: Box<dyn Query> =
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                let q = if *boost != 1.0 {
                    Box::new(BoostQuery::new(q, *boost))
                } else {
                    q
                };
                clauses.push((Occur::Should, q));
            }
        }
        if clauses.is_empty() {
            None
        } else {
            Some(Box::new(BooleanQuery::new(clauses)))
        }
    }

    /// The query's words, in order and adjacent, in the title. `None` for a
    /// query of fewer than two words (a one-word query gets its title weight
    /// from the per-word title match).
    fn title_phrase_query(&self, query: &str) -> Option<Box<dyn Query>> {
        let words = self.query_words(query);
        if words.len() < 2 {
            return None;
        }
        let terms: Vec<Term> = words
            .iter()
            .map(|w| Term::from_field_text(self.title_field, w))
            .collect();
        Some(Box::new(PhraseQuery::new(terms)))
    }

    /// Number of documents — auto-flushes pending writes first.
    pub fn len(&self) -> usize {
        if self.flush().is_err() {
            return 0;
        }
        if self.reader.reload().is_err() {
            return 0;
        }
        self.reader.searcher().num_docs() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build the canonical fields and schema. The `id` and `text` fields are
/// those of 0.4.3 and earlier; `title` and `url` were added in 0.4.4 and use
/// the same tokenizer as `text`. An index written before 0.4.4 has another
/// schema, so [`TextIndex::open_with`] recreates it empty and the memory
/// store refills it from its rows.
fn build_schema(tokenizer: TextTokenizer) -> (Fields, Schema) {
    let mut sb = Schema::builder();
    let id = sb.add_u64_field("id", STORED | INDEXED | FAST);
    let text_options: TextOptions = match tokenizer {
        TextTokenizer::Default => TEXT,
        TextTokenizer::Words => TEXT.set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(WORDS_TOKENIZER)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    };
    let text = sb.add_text_field("text", text_options.clone());
    let title = sb.add_text_field("title", text_options.clone());
    let url = sb.add_text_field("url", text_options);
    let schema = sb.build();
    (
        Fields {
            id,
            text,
            title,
            url,
        },
        schema,
    )
}

impl Default for TextIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TextIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextIndex")
            .field("len", &self.len())
            .field("commit_every", &self.commit_every)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_match_finds_only_relevant_doc() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund my order".into()).unwrap();
        idx.insert(2, "weather today".into()).unwrap();
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn bm25_ranks_higher_term_frequency_first() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund refund refund my order".into())
            .unwrap();
        idx.insert(2, "refund please".into()).unwrap();
        idx.insert(3, "weather forecast".into()).unwrap();
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 2, "doc 3 should be filtered out");
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let mut idx = TextIndex::new();
        idx.insert(1, "anything".into()).unwrap();
        assert!(idx.search("", 10).is_empty());
    }

    #[test]
    fn len_reflects_committed_count() {
        let mut idx = TextIndex::new();
        assert_eq!(idx.len(), 0);
        idx.insert(1, "one".into()).unwrap();
        idx.insert(2, "two".into()).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn batched_inserts_searchable_via_auto_flush() {
        // commit_every=10 means doc 1 isn't auto-committed by insert,
        // but search must still find it via auto-flush.
        let mut idx = TextIndex::with_commit_every(10);
        idx.insert(1, "single doc".into()).unwrap();
        let hits = idx.search("single", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn explicit_flush_makes_writes_durable() {
        let mut idx = TextIndex::with_commit_every(1000);
        for i in 0..50u64 {
            idx.insert(i, format!("doc {i}")).unwrap();
        }
        idx.flush().unwrap();
        assert_eq!(idx.len(), 50);
    }

    #[test]
    fn rebuild_drops_documents_not_in_survivors() {
        let mut idx = TextIndex::new();
        idx.insert(1, "refund my order".into()).unwrap();
        idx.insert(2, "weather today".into()).unwrap();
        idx.insert(3, "refund the payment".into()).unwrap();
        assert_eq!(idx.len(), 3);

        // Keep only {1}. Docs 2 and 3 are dropped from the postings.
        idx.rebuild(&[(1u64, "refund my order".to_string())])
            .unwrap();
        assert_eq!(idx.len(), 1);
        let hits = idx.search("refund", 10);
        assert_eq!(hits.len(), 1, "only survivor 1 should match refund");
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn disk_backed_index_persists_across_reopen() {
        let dir = std::env::temp_dir().join(format!(
            "duxx-tantivy-persist-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut idx = TextIndex::open(&dir).unwrap();
            idx.insert(1, "refund my order".into()).unwrap();
            idx.insert(2, "weather forecast".into()).unwrap();
            idx.flush().unwrap();
            assert_eq!(idx.len(), 2);
        } // drop closes index
        {
            let idx = TextIndex::open(&dir).unwrap();
            assert_eq!(idx.len(), 2, "data should survive reopen");
            let hits = idx.search("refund", 5);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "duxx-tantivy-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn words_of(text: &str) -> Vec<String> {
        let mut analyzer = words_analyzer();
        let mut stream = analyzer.token_stream(text);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn words_tokenizer_keeps_indian_words_whole() {
        assert_eq!(words_of("शिक्षा विभाग"), vec!["शिक्षा", "विभाग"]);
        assert_eq!(
            words_of("प्रधानमंत्री आवास योजना।"),
            vec!["प्रधानमंत्री", "आवास", "योजना"]
        );
        assert_eq!(words_of("ಕರ್ನಾಟಕ ಸರ್ಕಾರ"), vec!["ಕರ್ನಾಟಕ", "ಸರ್ಕಾರ"]);
        assert_eq!(words_of("കേരള സർക്കാർ"), vec!["കേരള", "സർക്കാർ"]);
    }

    #[test]
    fn words_tokenizer_splits_latin_text_as_before() {
        assert_eq!(
            words_of("Hello, happy tax-payer! RRB 2026"),
            vec!["hello", "happy", "tax", "payer", "rrb", "2026"]
        );
    }

    #[test]
    fn words_index_matches_whole_words_not_fragments() {
        let mut idx = TextIndex::in_ram_with(1, TextTokenizer::Words).unwrap();
        idx.insert(1, "शिक्षा विभाग बिहार".into()).unwrap();
        idx.insert(2, "weather today".into()).unwrap();
        assert_eq!(idx.search("शिक्षा", 10).first().map(|h| h.0), Some(1));
        assert!(idx.search("षा", 10).is_empty(), "a fragment is not a word");

        // The default tokenizer, unchanged, splits at the virama
        let mut old = TextIndex::in_ram_with(1, TextTokenizer::Default).unwrap();
        old.insert(1, "शिक्षा विभाग बिहार".into()).unwrap();
        assert_eq!(old.search("षा", 10).first().map(|h| h.0), Some(1));
    }

    #[test]
    fn query_that_is_not_valid_syntax_searches_its_words() {
        for tokenizer in [TextTokenizer::Default, TextTokenizer::Words] {
            let mut idx = TextIndex::in_ram_with(1, tokenizer).unwrap();
            idx.insert(1, "Thomas Markle arrested in Florida".into())
                .unwrap();
            idx.insert(2, "Faculty strength statistics".into()).unwrap();
            idx.insert(3, "weather today".into()).unwrap();
            let colon = idx.search("Thomas Markle arrested: Meghan", 10);
            assert_eq!(colon.first().map(|h| h.0), Some(1), "{tokenizer:?}");
            let bracket = idx.search("Faculty (strength statistics", 10);
            assert_eq!(bracket.first().map(|h| h.0), Some(2), "{tokenizer:?}");
            assert!(idx.search("((", 10).is_empty(), "{tokenizer:?}");
        }
    }

    #[test]
    fn reopening_with_another_tokenizer_recreates_the_index_empty() {
        let dir = temp_dir("retokenize");
        {
            let mut idx = TextIndex::open_with(&dir, 1, TextTokenizer::Default).unwrap();
            idx.insert(1, "शिक्षा विभाग".into()).unwrap();
            idx.insert(2, "weather today".into()).unwrap();
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.tokenizer(), TextTokenizer::Default);
        }
        {
            // Same tokenizer: the documents are kept
            let idx = TextIndex::open_with(&dir, 1, TextTokenizer::Default).unwrap();
            assert_eq!(idx.len(), 2);
        }
        {
            // Another tokenizer: emptied, for the caller to refill from its rows
            let mut idx = TextIndex::open_with(&dir, 1, TextTokenizer::Words).unwrap();
            assert_eq!(idx.len(), 0);
            assert_eq!(idx.tokenizer(), TextTokenizer::Words);
            idx.insert(1, "शिक्षा विभाग".into()).unwrap();
        }
        {
            let idx = TextIndex::open_with(&dir, 1, TextTokenizer::Words).unwrap();
            assert_eq!(idx.len(), 1);
            assert_eq!(idx.search("शिक्षा", 5).first().map(|h| h.0), Some(1));
            assert!(idx.search("षा", 5).is_empty());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // -------- 0.4.4: title and URL fields --------

    /// A page as Gaelxe writes it: URL, title, description, marker, body.
    fn page(url: &str, title: &str, description: &str, body: &str) -> String {
        format!("{url}\n{title}\n{description}\ngxm20261109Len\n{body}")
    }

    /// One site: its home page titled with the site's name, and two deep
    /// pages that mention the name often in their body but not in their title.
    fn wipro_site() -> Vec<(u64, String)> {
        vec![
            (
                1,
                page(
                    "https://www.wipro.com/",
                    "Wipro",
                    "Technology services and consulting company helping clients build better businesses",
                    "Explore our services, industries, insights and careers across the world.",
                ),
            ),
            (
                2,
                page(
                    "https://www.wipro.com/investors/annual-report/",
                    "Annual report 2025",
                    "Annual report",
                    "Wipro revenue grew. Wipro margins held. Wipro board met. Wipro paid a dividend.",
                ),
            ),
            (
                3,
                page(
                    "https://www.wipro.com/careers/india/",
                    "Careers in India",
                    "Jobs",
                    "Join Wipro. Wipro hires freshers. Wipro campus drives. Wipro offices.",
                ),
            ),
        ]
    }

    #[test]
    fn layout_gives_url_and_title() {
        let v2 = page(
            "https://www.india.gov.in/",
            "National Portal of India",
            "d",
            "b",
        );
        assert_eq!(
            DocFields::from_layout(&v2),
            DocFields {
                title: Some("National Portal of India".into()),
                url: Some("https://www.india.gov.in/".into()),
            }
        );
        // v1 (no marker line) has the same first two lines
        let v1 = "http://example.in/a\nExample title\ndescription\nbody";
        assert_eq!(
            DocFields::from_layout(v1).title.as_deref(),
            Some("Example title")
        );
        // A text that does not start with a URL has no fields
        assert!(DocFields::from_layout("I lost my wallet at the cafe").is_empty());
        assert!(DocFields::from_layout("https is a protocol\nnot a title").is_empty());
        assert!(DocFields::from_layout("https://\ntitle").is_empty());
        assert!(DocFields::from_layout("").is_empty());
        // Given fields win; missing ones come from the layout
        let given = DocFields {
            title: Some("Given".into()),
            url: None,
        };
        assert_eq!(
            given.or_layout(&v2),
            DocFields {
                title: Some("Given".into()),
                url: Some("https://www.india.gov.in/".into()),
            }
        );
    }

    #[test]
    fn url_is_split_into_words_without_scheme_and_www() {
        assert_eq!(
            url_words("https://www.digilocker.gov.in/"),
            "digilocker.gov.in/"
        );
        assert_eq!(
            url_words("http://nad.digilocker.gov.in"),
            "nad.digilocker.gov.in"
        );
        assert_eq!(url_words("WWW.example.com"), "example.com");
        let mut idx = TextIndex::in_ram_with(1, TextTokenizer::Words).unwrap();
        idx.set_field_boosts(FieldBoosts::default());
        idx.insert(1, page("https://www.digilocker.gov.in/", "Home", "d", "b"))
            .unwrap();
        idx.insert(2, "a page about something else".into()).unwrap();
        let hits = idx.field_search("digilocker", 10);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![1]);
        assert!(idx.field_search("www", 10).is_empty(), "www is dropped");
        assert!(
            idx.field_search("https", 10).is_empty(),
            "the scheme is dropped"
        );
    }

    #[test]
    fn title_match_outranks_body_mentions() {
        for tokenizer in [TextTokenizer::Default, TextTokenizer::Words] {
            let mut idx = TextIndex::in_ram_with(1, tokenizer).unwrap();
            for (id, text) in wipro_site() {
                idx.insert(id, text).unwrap();
            }
            // Fields on: the page titled "Wipro" comes first.
            idx.set_field_boosts(FieldBoosts::default());
            let on = idx.search("wipro", 10);
            assert_eq!(on.len(), 3, "{tokenizer:?}");
            assert_eq!(on[0].0, 1, "{tokenizer:?}: {on:?}");
            // Fields off (0.4.3 scoring): the pages that repeat the word win.
            idx.set_field_boosts(FieldBoosts::OFF);
            let off = idx.search("wipro", 10);
            assert_eq!(off.len(), 3, "{tokenizer:?}");
            assert_ne!(off[0].0, 1, "{tokenizer:?}: {off:?}");
            assert_eq!(off.last().map(|h| h.0), Some(1), "{tokenizer:?}: {off:?}");
        }
    }

    #[test]
    fn whole_query_in_the_title_earns_the_phrase_bonus() {
        let mut idx = TextIndex::in_ram_with(1, TextTokenizer::Default).unwrap();
        // Same words, same lengths; only doc 1 has them in the query's order.
        idx.insert(
            1,
            page("https://cars.example/a", "tata nexon price list", "d", "b"),
        )
        .unwrap();
        idx.insert(
            2,
            page("https://cars.example/b", "price list nexon tata", "d", "b"),
        )
        .unwrap();
        let score = |hits: &[(u64, f32)], id: u64| hits.iter().find(|h| h.0 == id).unwrap().1;

        idx.set_field_boosts(FieldBoosts::default());
        let on = idx.search("tata nexon price", 10);
        assert_eq!(on[0].0, 1, "{on:?}");
        assert!(score(&on, 1) > score(&on, 2) + 0.1, "{on:?}");

        idx.set_field_boosts(FieldBoosts {
            title_phrase: 0.0,
            ..FieldBoosts::default()
        });
        let no_phrase = idx.search("tata nexon price", 10);
        assert!(
            (score(&no_phrase, 1) - score(&no_phrase, 2)).abs() < 1e-4,
            "{no_phrase:?}"
        );
        // A one-word query has no phrase
        assert!(idx.title_phrase_query("nexon").is_none());
    }

    #[test]
    fn documents_without_fields_score_exactly_as_before() {
        let texts = [
            "refund refund my order",
            "refund please",
            "weather forecast for Pune",
            "Thomas Markle arrested in Florida",
        ];
        for tokenizer in [TextTokenizer::Default, TextTokenizer::Words] {
            let mut idx = TextIndex::in_ram_with(1, tokenizer).unwrap();
            for (i, t) in texts.iter().enumerate() {
                idx.insert(i as u64 + 1, t.to_string()).unwrap();
            }
            for q in [
                "refund",
                "refund order",
                "weather Pune",
                "arrested: Markle",
                "\"refund my\"",
            ] {
                idx.set_field_boosts(FieldBoosts::OFF);
                let before = idx.search(q, 10);
                idx.set_field_boosts(FieldBoosts::default());
                let after = idx.search(q, 10);
                assert!(!before.is_empty(), "{tokenizer:?} {q}");
                assert_eq!(before.len(), after.len(), "{tokenizer:?} {q}");
                for (b, a) in before.iter().zip(&after) {
                    assert_eq!(b.0, a.0, "{tokenizer:?} {q}");
                    assert!(
                        (b.1 - a.1).abs() < 1e-6,
                        "{tokenizer:?} {q}: {b:?} vs {a:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn given_fields_are_indexed_and_kept_by_rebuild() {
        let mut idx = TextIndex::in_ram_with(1, TextTokenizer::Default).unwrap();
        idx.set_field_boosts(FieldBoosts::default());
        let fields = DocFields {
            title: Some("Digilocker login".into()),
            url: Some("https://www.digilocker.gov.in/".into()),
        };
        idx.insert_with(1, "Sign in to your documents".into(), &fields)
            .unwrap();
        idx.insert(2, "unrelated text".into()).unwrap();
        assert_eq!(idx.field_search("login", 10).first().map(|h| h.0), Some(1));
        idx.rebuild_with(&[(1, "Sign in to your documents".into(), fields)])
            .unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(
            idx.field_search("digilocker", 10).first().map(|h| h.0),
            Some(1)
        );
    }

    #[test]
    fn search_until_stops_at_the_deadline() {
        // One segment per document (commit_every = 1, no merges this small).
        let mut idx = TextIndex::in_ram_with(1, TextTokenizer::Default).unwrap();
        for i in 0..20u64 {
            idx.insert(i, format!("monsoon rainfall report {i}"))
                .unwrap();
        }
        let full = idx.search("monsoon", 50);
        assert_eq!(full.len(), 20);

        // Deadline already passed: nothing is searched, and it says so.
        let found = Mutex::new(Vec::new());
        let complete = idx.search_until("monsoon", 50, Instant::now(), &found);
        assert!(!complete);
        assert!(found.lock().is_empty());

        // Time to spare: the same hits as a full search.
        let found = Mutex::new(Vec::new());
        let later = Instant::now() + std::time::Duration::from_secs(30);
        assert!(idx.search_until("monsoon", 50, later, &found));
        let mut got: Vec<u64> = found.lock().iter().map(|h| h.0).collect();
        let mut want: Vec<u64> = full.iter().map(|h| h.0).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn index_from_before_the_title_field_is_recreated() {
        // An index written by 0.4.3: only `id` and `text`.
        let dir = temp_dir("old-schema");
        {
            let mut sb = Schema::builder();
            let id = sb.add_u64_field("id", STORED | INDEXED | FAST);
            let text = sb.add_text_field("text", TEXT);
            std::fs::create_dir_all(&dir).unwrap();
            let index = Index::create_in_dir(&dir, sb.build()).unwrap();
            let mut w: IndexWriter = index.writer_with_num_threads(1, 20_000_000).unwrap();
            w.add_document(doc!(id => 1u64, text => "old row")).unwrap();
            w.commit().unwrap();
        }
        let mut idx = TextIndex::open_with(&dir, 1, TextTokenizer::Default).unwrap();
        assert_eq!(
            idx.len(),
            0,
            "emptied, for the store to refill from its rows"
        );
        idx.insert(1, page("https://www.wipro.com/", "Wipro", "d", "b"))
            .unwrap();
        drop(idx);
        let idx = TextIndex::open_with(&dir, 1, TextTokenizer::Default).unwrap();
        assert_eq!(idx.len(), 1, "the new schema opens as it is");
        std::fs::remove_dir_all(&dir).ok();
    }
}
