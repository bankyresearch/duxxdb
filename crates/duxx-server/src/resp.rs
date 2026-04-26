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
    if buf.is_empty() {
        return Ok(None);
    }
    let mut cursor = 0;
    match parse_value(&buf[..], &mut cursor) {
        Ok(v) => {
            buf.advance(cursor);
            Ok(Some(v))
        }
        Err(ParseError::Incomplete) => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_value(data: &[u8], cursor: &mut usize) -> Result<RespValue, ParseError> {
    if *cursor >= data.len() {
        return Err(ParseError::Incomplete);
    }
    let prefix = data[*cursor];
    match prefix {
        b'+' => {
            *cursor += 1;
            let line = read_line(data, cursor)?;
            Ok(RespValue::SimpleString(line.to_string()))
        }
        b'-' => {
            *cursor += 1;
            let line = read_line(data, cursor)?;
            Ok(RespValue::Error(line.to_string()))
        }
        b':' => {
            *cursor += 1;
            let line = read_line(data, cursor)?;
            let n: i64 = line.parse().map_err(|_| ParseError::Bad(format!("bad int {line:?}")))?;
            Ok(RespValue::Integer(n))
        }
        b'$' => {
            *cursor += 1;
            let line = read_line(data, cursor)?;
            let len: i64 = line.parse().map_err(|_| ParseError::Bad(format!("bad bulk len {line:?}")))?;
            if len < 0 {
                return Ok(RespValue::Null);
            }
            let len = len as usize;
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
            let line = read_line(data, cursor)?;
            let len: i64 = line.parse().map_err(|_| ParseError::Bad(format!("bad array len {line:?}")))?;
            if len < 0 {
                return Ok(RespValue::Null);
            }
            let len = len as usize;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(parse_value(data, cursor)?);
            }
            Ok(RespValue::Array(items))
        }
        _ => {
            // Inline command: no prefix. Read the rest of the line and
            // split on whitespace into bulk strings.
            let line = read_line(data, cursor)?;
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
fn read_line<'a>(data: &'a [u8], cursor: &mut usize) -> Result<&'a str, ParseError> {
    let start = *cursor;
    let mut i = start;
    while i + 1 < data.len() {
        if data[i] == b'\r' && data[i + 1] == b'\n' {
            let line = str::from_utf8(&data[start..i])
                .map_err(|e| ParseError::Bad(format!("non-utf8 line: {e}")))?;
            *cursor = i + 2;
            return Ok(line);
        }
        i += 1;
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
}
