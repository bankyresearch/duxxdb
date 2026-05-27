//! RESP2/3 wire codec.
//!
//! RESP (REdis Serialization Protocol) is line-oriented and trivially
//! parseable. We support the RESP2 surface that every Valkey/Redis
//! client knows how to talk:
//!
//! | Prefix | Type            | Example                      |
//! |--------|-----------------|------------------------------|
//! | `+`    | Simple String   | `+PONG\r\n`                  |
//! | `-`    | Error           | `-ERR bad cmd\r\n`           |
//! | `:`    | Integer         | `:42\r\n`                    |
//! | `$`    | Bulk String     | `$5\r\nhello\r\n` / `$-1\r\n`|
//! | `*`    | Array           | `*2\r\n$3\r\nfoo\r\n:7\r\n`  |
//!
//! Inline commands (no `*` framing — e.g. `PING\r\n`) are also parsed
//! so `telnet`/`nc` users can poke the server.

use bytes::{Buf, BytesMut};
use std::str;

/// Maximum bytes accepted for one RESP bulk string.
pub const MAX_BULK_BYTES: usize = 16 * 1024 * 1024;
/// Maximum items accepted in one RESP array.
pub const MAX_ARRAY_ITEMS: usize = 4096;
/// Maximum bytes accepted in a single RESP line before `\r\n`.
pub const MAX_LINE_BYTES: usize = 64 * 1024;
/// Maximum buffered inbound bytes per connection before the server
/// closes it. This caps slowloris-style partial frames and excessive
/// pipelining before authentication.
pub const MAX_INPUT_BUFFER_BYTES: usize = 32 * 1024 * 1024;
/// Maximum recursive nesting accepted for RESP arrays.
pub const MAX_NESTING_DEPTH: usize = 64;

/// Runtime limits for RESP parsing and per-connection buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RespLimits {
    pub max_bulk_bytes: usize,
    pub max_array_items: usize,
    pub max_line_bytes: usize,
    pub max_input_buffer_bytes: usize,
    pub max_nesting_depth: usize,
}

impl RespLimits {
    pub const fn new(
        max_bulk_bytes: usize,
        max_array_items: usize,
        max_line_bytes: usize,
        max_input_buffer_bytes: usize,
        max_nesting_depth: usize,
    ) -> Self {
        Self {
            max_bulk_bytes,
            max_array_items,
            max_line_bytes,
            max_input_buffer_bytes,
            max_nesting_depth,
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        Self {
            max_bulk_bytes: read_limit_env("DUXX_MAX_BULK_BYTES", MAX_BULK_BYTES)?,
            max_array_items: read_limit_env("DUXX_MAX_ARRAY_ITEMS", MAX_ARRAY_ITEMS)?,
            max_line_bytes: read_limit_env("DUXX_MAX_LINE_BYTES", MAX_LINE_BYTES)?,
            max_input_buffer_bytes: read_limit_env(
                "DUXX_MAX_INPUT_BUFFER_BYTES",
                MAX_INPUT_BUFFER_BYTES,
            )?,
            max_nesting_depth: read_limit_env("DUXX_MAX_NESTING_DEPTH", MAX_NESTING_DEPTH)?,
        }
        .validate()
    }

    pub fn validate(self) -> anyhow::Result<Self> {
        validate_limit("max_bulk_bytes", self.max_bulk_bytes)?;
        validate_limit("max_array_items", self.max_array_items)?;
        validate_limit("max_line_bytes", self.max_line_bytes)?;
        validate_limit("max_input_buffer_bytes", self.max_input_buffer_bytes)?;
        validate_limit("max_nesting_depth", self.max_nesting_depth)?;
        Ok(self)
    }
}

impl Default for RespLimits {
    fn default() -> Self {
        Self::new(
            MAX_BULK_BYTES,
            MAX_ARRAY_ITEMS,
            MAX_LINE_BYTES,
            MAX_INPUT_BUFFER_BYTES,
            MAX_NESTING_DEPTH,
        )
    }
}

fn read_limit_env(name: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .trim()
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("{name} must be a positive integer: {e}")),
        Ok(_) => anyhow::bail!("{name} must not be empty"),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => anyhow::bail!("reading {name}: {e}"),
    }
}

fn validate_limit(name: &str, value: usize) -> anyhow::Result<()> {
    if value == 0 {
        anyhow::bail!("{name} must be greater than zero");
    }
    Ok(())
}

/// One RESP value.
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Vec<u8>),
    Array(Vec<RespValue>),
    Null,
}

impl RespValue {
    /// Convenience: a non-null bulk string from a `&str`.
    pub fn bulk(s: impl Into<String>) -> Self {
        let s = s.into();
        RespValue::BulkString(s.into_bytes())
    }

    /// Append the wire-format encoding to `out`.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        match self {
            RespValue::SimpleString(s) => {
                out.push(b'+');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            RespValue::Error(s) => {
                out.push(b'-');
                out.extend_from_slice(s.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            RespValue::Integer(i) => {
                out.push(b':');
                out.extend_from_slice(i.to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            RespValue::BulkString(bytes) => {
                out.push(b'$');
                out.extend_from_slice(bytes.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                out.extend_from_slice(bytes);
                out.extend_from_slice(b"\r\n");
            }
            RespValue::Array(items) => {
                out.push(b'*');
                out.extend_from_slice(items.len().to_string().as_bytes());
                out.extend_from_slice(b"\r\n");
                for v in items {
                    v.write_to(out);
                }
            }
            RespValue::Null => {
                // RESP2 null bulk string.
                out.extend_from_slice(b"$-1\r\n");
            }
        }
    }

    /// Encode as a fresh `Vec<u8>` — convenient for tests.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.write_to(&mut v);
        v
    }
}

/// Parse errors.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// Buffer doesn't yet contain a full message — read more bytes and retry.
    #[error("incomplete")]
    Incomplete,
    /// Wire format is malformed.
    #[error("protocol error: {0}")]
    Bad(String),
}

/// Try to consume one RESP value from `buf`. On success the consumed
/// bytes are removed from `buf`.
pub fn parse(buf: &mut BytesMut) -> Result<Option<RespValue>, ParseError> {
    parse_with_limits(buf, RespLimits::default())
}

/// Try to consume one RESP value from `buf` using explicit runtime
/// limits. On success the consumed bytes are removed from `buf`.
pub fn parse_with_limits(
    buf: &mut BytesMut,
    limits: RespLimits,
) -> Result<Option<RespValue>, ParseError> {
    if buf.is_empty() {
        return Ok(None);
    }
    let mut cursor = 0;
    match parse_value(&buf[..], &mut cursor, &limits, 0) {
        Ok(v) => {
            buf.advance(cursor);
            Ok(Some(v))
        }
        Err(ParseError::Incomplete) => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_value(
    data: &[u8],
    cursor: &mut usize,
    limits: &RespLimits,
    depth: usize,
) -> Result<RespValue, ParseError> {
    if depth > limits.max_nesting_depth {
        return Err(ParseError::Bad(format!(
            "RESP nesting too deep: > {}",
            limits.max_nesting_depth
        )));
    }
    if *cursor >= data.len() {
        return Err(ParseError::Incomplete);
    }
    let prefix = data[*cursor];
    match prefix {
        b'+' => {
            *cursor += 1;
            let line = read_line(data, cursor, limits)?;
            Ok(RespValue::SimpleString(line.to_string()))
        }
        b'-' => {
            *cursor += 1;
            let line = read_line(data, cursor, limits)?;
            Ok(RespValue::Error(line.to_string()))
        }
        b':' => {
            *cursor += 1;
            let line = read_line(data, cursor, limits)?;
            let n: i64 = line
                .parse()
                .map_err(|_| ParseError::Bad(format!("bad int {line:?}")))?;
            Ok(RespValue::Integer(n))
        }
        b'$' => {
            *cursor += 1;
            let line = read_line(data, cursor, limits)?;
            let len: i64 = line
                .parse()
                .map_err(|_| ParseError::Bad(format!("bad bulk len {line:?}")))?;
            if len < 0 {
                return Ok(RespValue::Null);
            }
            let len = len as usize;
            if len > limits.max_bulk_bytes {
                return Err(ParseError::Bad(format!(
                    "bulk string too large: {len} bytes > {}",
                    limits.max_bulk_bytes
                )));
            }
            if *cursor + len + 2 > data.len() {
                return Err(ParseError::Incomplete);
            }
            let bytes = data[*cursor..*cursor + len].to_vec();
            *cursor += len;
            if &data[*cursor..*cursor + 2] != b"\r\n" {
                return Err(ParseError::Bad("missing \\r\\n after bulk string".into()));
            }
            *cursor += 2;
            Ok(RespValue::BulkString(bytes))
        }
        b'*' => {
            *cursor += 1;
            let line = read_line(data, cursor, limits)?;
            let len: i64 = line
                .parse()
                .map_err(|_| ParseError::Bad(format!("bad array len {line:?}")))?;
            if len < 0 {
                return Ok(RespValue::Null);
            }
            let len = len as usize;
            if len > limits.max_array_items {
                return Err(ParseError::Bad(format!(
                    "array too large: {len} items > {}",
                    limits.max_array_items
                )));
            }
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(parse_value(data, cursor, limits, depth + 1)?);
            }
            Ok(RespValue::Array(items))
        }
        _ => {
            // Inline command: no prefix. Read the rest of the line and
            // split on whitespace into bulk strings.
            let line = read_line(data, cursor, limits)?;
            let items = line
                .split_whitespace()
                .map(|tok| RespValue::bulk(tok.to_string()))
                .collect();
            Ok(RespValue::Array(items))
        }
    }
}

/// Read up to `\r\n` and advance the cursor past it. Returns the line
/// content (without the trailer).
fn read_line<'a>(
    data: &'a [u8],
    cursor: &mut usize,
    limits: &RespLimits,
) -> Result<&'a str, ParseError> {
    let start = *cursor;
    let mut i = start;
    while i + 1 < data.len() {
        if i - start > limits.max_line_bytes {
            return Err(ParseError::Bad(format!(
                "line too large: > {} bytes",
                limits.max_line_bytes
            )));
        }
        if data[i] == b'\r' && data[i + 1] == b'\n' {
            let line = str::from_utf8(&data[start..i])
                .map_err(|e| ParseError::Bad(format!("non-utf8 line: {e}")))?;
            *cursor = i + 2;
            return Ok(line);
        }
        i += 1;
    }
    if data.len().saturating_sub(start) > limits.max_line_bytes {
        return Err(ParseError::Bad(format!(
            "line too large: > {} bytes",
            limits.max_line_bytes
        )));
    }
    Err(ParseError::Incomplete)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: RespValue) {
        let bytes = v.to_bytes();
        let mut buf = BytesMut::from(&bytes[..]);
        let parsed = parse(&mut buf).unwrap().unwrap();
        assert_eq!(parsed, v);
        assert!(buf.is_empty());
    }

    #[test]
    fn simple_string_roundtrip() {
        roundtrip(RespValue::SimpleString("OK".into()));
        roundtrip(RespValue::SimpleString("PONG".into()));
    }

    #[test]
    fn integer_roundtrip() {
        roundtrip(RespValue::Integer(0));
        roundtrip(RespValue::Integer(42));
        roundtrip(RespValue::Integer(-7));
    }

    #[test]
    fn bulk_string_roundtrip() {
        roundtrip(RespValue::bulk("hello world"));
        roundtrip(RespValue::BulkString(vec![0u8, 1, 2, 3]));
    }

    #[test]
    fn null_bulk_roundtrip() {
        let mut buf = BytesMut::from(&b"$-1\r\n"[..]);
        let v = parse(&mut buf).unwrap().unwrap();
        assert_eq!(v, RespValue::Null);
    }

    #[test]
    fn array_roundtrip() {
        roundtrip(RespValue::Array(vec![
            RespValue::bulk("SET"),
            RespValue::bulk("k"),
            RespValue::bulk("v"),
        ]));
    }

    #[test]
    fn incomplete_returns_none() {
        let mut buf = BytesMut::from(&b"$5\r\nhel"[..]); // not yet 5 bytes
        assert!(parse(&mut buf).unwrap().is_none());
    }

    #[test]
    fn inline_command_parses_as_array() {
        let mut buf = BytesMut::from(&b"PING\r\n"[..]);
        let v = parse(&mut buf).unwrap().unwrap();
        assert_eq!(v, RespValue::Array(vec![RespValue::bulk("PING")]));
    }

    #[test]
    fn inline_with_args() {
        let mut buf = BytesMut::from(&b"REMEMBER alice hello\r\n"[..]);
        let v = parse(&mut buf).unwrap().unwrap();
        assert_eq!(
            v,
            RespValue::Array(vec![
                RespValue::bulk("REMEMBER"),
                RespValue::bulk("alice"),
                RespValue::bulk("hello"),
            ])
        );
    }

    #[test]
    fn pipelined_commands() {
        let mut buf = BytesMut::from(&b"+OK\r\n+PONG\r\n"[..]);
        let a = parse(&mut buf).unwrap().unwrap();
        let b = parse(&mut buf).unwrap().unwrap();
        assert_eq!(a, RespValue::SimpleString("OK".into()));
        assert_eq!(b, RespValue::SimpleString("PONG".into()));
        assert!(buf.is_empty());
    }

    #[test]
    fn oversized_bulk_is_rejected_before_body_allocation() {
        let bytes = format!("${}\r\n", MAX_BULK_BYTES + 1);
        let mut buf = BytesMut::from(bytes.as_bytes());
        let err = parse(&mut buf).unwrap_err().to_string();
        assert!(err.contains("bulk string too large"), "{err}");
    }

    #[test]
    fn oversized_array_is_rejected_before_allocation() {
        let bytes = format!("*{}\r\n", MAX_ARRAY_ITEMS + 1);
        let mut buf = BytesMut::from(bytes.as_bytes());
        let err = parse(&mut buf).unwrap_err().to_string();
        assert!(err.contains("array too large"), "{err}");
    }

    #[test]
    fn oversized_line_is_rejected() {
        let line = "x".repeat(MAX_LINE_BYTES + 2);
        let mut buf = BytesMut::from(line.as_bytes());
        let err = parse(&mut buf).unwrap_err().to_string();
        assert!(err.contains("line too large"), "{err}");
    }

    #[test]
    fn custom_limits_are_enforced() {
        let limits = RespLimits::new(3, MAX_ARRAY_ITEMS, MAX_LINE_BYTES, 128, MAX_NESTING_DEPTH);
        let mut buf = BytesMut::from(&b"$4\r\ntest\r\n"[..]);
        let err = parse_with_limits(&mut buf, limits).unwrap_err().to_string();
        assert!(err.contains("bulk string too large"), "{err}");
    }

    #[test]
    fn nested_arrays_are_limited() {
        let limits = RespLimits::new(
            MAX_BULK_BYTES,
            MAX_ARRAY_ITEMS,
            MAX_LINE_BYTES,
            MAX_INPUT_BUFFER_BYTES,
            1,
        );
        let mut buf = BytesMut::from(&b"*1\r\n*1\r\n*0\r\n"[..]);
        let err = parse_with_limits(&mut buf, limits).unwrap_err().to_string();
        assert!(err.contains("RESP nesting too deep"), "{err}");
    }
}
