//! Tantivy-backed BM25 text index with batched commits.
//!
//! Phase 2.5: instead of committing tantivy on every `insert` (slow:
//! ~4 ms each), we commit every `commit_every` inserts. Reads (`search`,
//! `len`) auto-flush any pending writes so the "insert-then-search"
//! contract still holds.
//!
//! Defaults to `commit_every = 100`. Override at construction with
//! [`TextIndex::with_commit_every`].

use duxx_core::{Error, Result};
use parking_lot::Mutex;
use std::path::Path;
use std::str::CharIndices;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    doc,
    query::{BooleanQuery, Occur, Query, QueryParser, TermQuery},
    schema::{
        Field, IndexRecordOption, Schema, TextFieldIndexing, Value, FAST, INDEXED, STORED, TEXT,
    },
    tokenizer::{LowerCaser, RemoveLongFilter, TextAnalyzer, Token, TokenStream, Tokenizer},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, TantivyError, Term,
};

/// Default insert count between forced commits.
pub const DEFAULT_COMMIT_EVERY: usize = 100;

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

/// BM25 full-text index, in-memory, Send + Sync.
pub struct TextIndex {
    id_field: Field,
    text_field: Field,
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
        let (id_field, text_field, schema) = build_schema(tokenizer);
        let index = Index::create_in_ram(schema);
        Self::from_index(index, id_field, text_field, commit_every)
    }

    /// Open a disk-backed tantivy index at `dir`. The directory is
    /// created if missing. Reopening an existing dir restores the
    /// previously written state — the writer queue is empty, so the
    /// caller can immediately commit new docs without rebuild.
    ///
    /// The tokenizer comes from `DUXX_TEXT_TOKENIZER`. When the index on
    /// disk was built with another tokenizer, it is recreated empty: the
    /// text index is derived from the rows, and `MemoryStore::open_at`
    /// refills it from the row store because its count no longer matches.
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
        let (id_field, text_field, schema) = build_schema(tokenizer);
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
        Self::from_index(index, id_field, text_field, commit_every)
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

    fn from_index(
        index: Index,
        id_field: Field,
        text_field: Field,
        commit_every: usize,
    ) -> Result<Self> {
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
            id_field,
            text_field,
            index,
            writer: Arc::new(Mutex::new(writer)),
            reader,
            pending: AtomicUsize::new(0),
            commit_every: commit_every.max(1),
        })
    }

    /// Insert a document. Commits if the threshold is reached.
    pub fn insert(&mut self, id: u64, text: String) -> Result<()> {
        let mut w = self.writer.lock();
        w.add_document(doc!(self.id_field => id, self.text_field => text))
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
        let mut w = self.writer.lock();
        // Flush any batched-but-uncommitted inserts into segments first, so
        // `delete_all_documents` (which only targets committed docs) actually
        // removes them. Without this, pending docs survive the rebuild.
        w.commit()
            .map_err(|e| Error::Index(format!("commit (flush pending): {e}")))?;
        w.delete_all_documents()
            .map_err(|e| Error::Index(format!("delete_all_documents: {e}")))?;
        for (id, text) in survivors {
            w.add_document(doc!(self.id_field => *id, self.text_field => text.clone()))
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

    /// BM25 search; auto-flushes pending writes first.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        if let Err(e) = self.flush() {
            tracing::warn!("flush before search failed: {e}");
            return Vec::new();
        }
        if let Err(e) = self.reader.reload() {
            tracing::warn!("tantivy reader reload failed: {e}");
            return Vec::new();
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        // Plain text often is not valid query syntax: a title with a colon
        // ("arrested: …" reads as a field name) or an unbalanced bracket or
        // quote failed to parse, and the text side of recall returned
        // nothing. Such a query is searched as its words instead.
        let parsed: Box<dyn Query> = match parser.parse_query(query) {
            Ok(q) => q,
            Err(e) => {
                tracing::debug!("query parse failed ({e}); searching its words");
                match self.words_query(query) {
                    Some(q) => q,
                    None => return Vec::new(),
                }
            }
        };
        let docs = match searcher.search(&parsed, &TopDocs::with_limit(k)) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("tantivy search failed: {e}");
                return Vec::new();
            }
        };
        docs.into_iter()
            .filter_map(|(score, addr)| {
                let d: TantivyDocument = searcher.doc(addr).ok()?;
                let v = d.get_first(self.id_field)?;
                let id = v.as_u64()?;
                Some((id, score))
            })
            .collect()
    }

    /// Any of the query's words, split by the field's own analyzer, scored
    /// by BM25: what the query parser makes of plain words, without its
    /// syntax. `None` when the query has no words.
    fn words_query(&self, query: &str) -> Option<Box<dyn Query>> {
        let mut analyzer = self.index.tokenizer_for_field(self.text_field).ok()?;
        let mut stream = analyzer.token_stream(query);
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        while stream.advance() {
            let term = Term::from_field_text(self.text_field, &stream.token().text);
            clauses.push((
                Occur::Should,
                Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)),
            ));
        }
        if clauses.is_empty() {
            None
        } else {
            Some(Box::new(BooleanQuery::new(clauses)))
        }
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

/// Build the canonical (id_field, text_field, schema) triple. Matches
/// the on-disk schema so re-opens succeed. With [`TextTokenizer::Default`]
/// it is exactly the 0.4.2 schema, so existing indexes open unchanged.
fn build_schema(tokenizer: TextTokenizer) -> (Field, Field, Schema) {
    let mut sb = Schema::builder();
    let id_field = sb.add_u64_field("id", STORED | INDEXED | FAST);
    let text_options = match tokenizer {
        TextTokenizer::Default => TEXT,
        TextTokenizer::Words => TEXT.set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(WORDS_TOKENIZER)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        ),
    };
    let text_field = sb.add_text_field("text", text_options);
    let schema = sb.build();
    (id_field, text_field, schema)
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
}
