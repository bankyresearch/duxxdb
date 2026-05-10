//! Tiny glob matcher for `PSUBSCRIBE` patterns.
//!
//! Supports the same subset Redis uses for `PSUBSCRIBE` / `KEYS`:
//!
//! - `*`    matches any sequence (including empty)
//! - `?`    matches any single character
//! - `[…]`  is **not** supported (Redis supports it; we don't yet)
//! - `\\c`  literal escape — pass through `c`
//! - everything else is a literal char
//!
//! Linear-time backtracking matcher. No regex compile step, no allocation.

/// `true` iff `text` matches the glob `pattern`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_bytes(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_bytes(mut pat: &[u8], mut s: &[u8]) -> bool {
    // Backtrack indices for the most recent `*` we can re-expand.
    let mut star_pat: Option<&[u8]> = None;
    let mut star_s: &[u8] = &[];

    loop {
        match (pat.first(), s.first()) {
            (Some(b'\\'), _) if pat.len() >= 2 => {
                // Escaped literal: \X must match X.
                let want = pat[1];
                if let Some(&got) = s.first() {
                    if got == want {
                        pat = &pat[2..];
                        s = &s[1..];
                        continue;
                    }
                }
                // Fall through to backtrack.
            }
            (Some(b'?'), Some(_)) => {
                pat = &pat[1..];
                s = &s[1..];
                continue;
            }
            (Some(b'*'), _) => {
                // Skip runs of *s.
                while let Some(b'*') = pat.first() {
                    pat = &pat[1..];
                }
                if pat.is_empty() {
                    return true;
                }
                star_pat = Some(pat);
                star_s = s;
                continue;
            }
            (Some(&pc), Some(&sc)) if pc == sc => {
                pat = &pat[1..];
                s = &s[1..];
                continue;
            }
            (None, None) => return true,
            _ => {}
        }

        // Try to extend the most recent star, if any.
        if let Some(sp) = star_pat {
            if !star_s.is_empty() {
                star_s = &star_s[1..];
                pat = sp;
                s = star_s;
                continue;
            }
        }
        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::glob_match as m;

    #[test]
    fn literal() {
        assert!(m("abc", "abc"));
        assert!(!m("abc", "abd"));
        assert!(!m("abc", "ab"));
    }

    #[test]
    fn star() {
        assert!(m("*", ""));
        assert!(m("*", "anything"));
        assert!(m("a*", "a"));
        assert!(m("a*", "abc"));
        assert!(!m("a*", ""));
        assert!(m("*c", "abc"));
        assert!(m("a*c", "abc"));
        assert!(m("a*c", "axxxc"));
        assert!(!m("a*c", "axxxd"));
    }

    #[test]
    fn question() {
        assert!(m("a?c", "abc"));
        assert!(m("a?c", "axc"));
        assert!(!m("a?c", "ac"));
        assert!(!m("a?c", "abbc"));
    }

    #[test]
    fn escape() {
        assert!(m(r"a\*b", "a*b"));
        assert!(!m(r"a\*b", "axb"));
    }

    #[test]
    fn redis_psubscribe_examples() {
        assert!(m("memory.*", "memory.alice"));
        assert!(m("memory.*", "memory."));
        assert!(!m("memory.*", "memory"));
        assert!(m("memory.a*", "memory.alice"));
        assert!(!m("memory.a*", "memory.bob"));
        assert!(m("*", "anything"));
    }
}
