#[cfg(test)]
use std::borrow::Cow;

use crate::error::{Error, ErrorCode, Result};
use crate::json::{JsonBuilder, JsonValue};

mod normalized;
mod query_terms;
mod visit;
pub(crate) use query_terms::{query_terms_in, QueryTerms};
pub(crate) use visit::{visit_tokens_in, visit_tokens_in_reclaiming};

#[cfg(test)]
thread_local! {
    pub(crate) static TOKEN_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The immutable tokenizer identity shared by storage, import and query outputs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum TokenizerConfig {
    #[default]
    DefaultEn,
}

impl TokenizerConfig {
    pub fn default_en() -> Self {
        Self::DefaultEn
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "default_en" => Ok(Self::DefaultEn),
            _ => Err(
                Error::new(ErrorCode::InvalidJsonlSchema, "unsupported tokenizer name")
                    .with_detail("field", "tokenizer.name"),
            ),
        }
    }

    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("name", builder.string("default_en")?),
            ("version", builder.string("0.1")?),
            ("profile", builder.string("latin_code_default")?),
            ("language_hint", builder.string("en")?),
            (
                "config",
                builder.object([
                    ("case_fold", builder.string("lower")?),
                    ("split", builder.string("unicode_punctuation")?),
                    ("stop_words", builder.string("default_en")?),
                ])?,
            ),
        ])
    }

    pub fn from_json(value: &JsonValue) -> Result<Self> {
        value
            .validate_kind(crate::json::JsonInputKind::CollectionConfig)
            .map_err(|err| err.with_detail("field", "tokenizer"))?;
        if value != &Self::default().to_json() {
            return Err(Error::new(
                ErrorCode::InvalidJsonlSchema,
                "tokenizer must be the complete immutable default_en 0.1 object",
            )
            .with_detail("field", "tokenizer")
            .with_detail("reason", "unsupported_or_incomplete_tokenizer"));
        }
        Ok(Self::DefaultEn)
    }
}

pub const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with",
];

#[cfg(test)]
pub(crate) fn tokenize(input: &str) -> Vec<String> {
    tokens(input).map(Cow::into_owned).collect()
}

#[cfg(test)]
pub(crate) fn tokens(input: &str) -> impl Iterator<Item = Cow<'_, str>> + '_ {
    symbols(input).flat_map(|symbol| {
        let mut parts = symbol_parts(symbol);
        let first = parts.next();
        let whole = normalized_token(symbol);
        if first.is_none() || first == Some(symbol) {
            return SymbolTokens::Single(whole.into_iter());
        }
        let mut unique = Vec::new();
        if let Some(token) = whole {
            unique.push(token);
        }
        for part in first.into_iter().chain(parts) {
            if let Some(token) = normalized_token(part) {
                if !unique.contains(&token) {
                    unique.push(token);
                }
            }
        }
        SymbolTokens::Split(unique.into_iter())
    })
}

// Query consumers deduplicate globally; staging a per-symbol set is unnecessary.
#[cfg(test)]
pub(crate) fn query_tokens(input: &str) -> impl Iterator<Item = Cow<'_, str>> + '_ {
    symbols(input)
        .flat_map(|symbol| {
            std::iter::once(symbol).chain(symbol_parts(symbol).filter(move |part| *part != symbol))
        })
        .filter_map(normalized_token)
}

fn symbols(input: &str) -> impl Iterator<Item = &str> {
    input
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|symbol| !symbol.is_empty())
}

#[cfg(test)]
enum SymbolTokens<'a> {
    Single(std::option::IntoIter<Cow<'a, str>>),
    Split(std::vec::IntoIter<Cow<'a, str>>),
}

#[cfg(test)]
impl<'a> Iterator for SymbolTokens<'a> {
    type Item = Cow<'a, str>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Single(tokens) => tokens.next(),
            Self::Split(tokens) => tokens.next(),
        }
    }
}

#[cfg(test)]
fn normalized_token(token: &str) -> Option<Cow<'_, str>> {
    let unchanged = if token.is_ascii() {
        !token.bytes().any(|byte| byte.is_ascii_uppercase())
    } else {
        token
            .chars()
            .all(|ch| ch.to_lowercase().eq(std::iter::once(ch)))
    };
    // Keep contextual Unicode casing (notably final sigma) in the standard library.
    let token = if unchanged {
        Cow::Borrowed(token)
    } else {
        Cow::Owned(token.to_lowercase())
    };
    (!token.is_empty() && !STOP_WORDS.contains(&token.as_ref())).then_some(token)
}

fn symbol_parts(symbol: &str) -> impl Iterator<Item = &str> {
    symbol
        .split('_')
        .filter(|part| !part.is_empty())
        .flat_map(|part| {
            let mut chars = part.char_indices().peekable();
            let mut previous = chars.next().expect("nonempty symbol part").1;
            let mut start = 0;
            std::iter::from_fn(move || {
                if start == part.len() {
                    return None;
                }
                while let Some((offset, current)) = chars.next() {
                    let boundary = (previous.is_lowercase() && current.is_uppercase())
                        || (previous.is_uppercase()
                            && current.is_uppercase()
                            && chars.peek().is_some_and(|(_, ch)| ch.is_lowercase()));
                    previous = current;
                    if boundary {
                        let token = &part[start..offset];
                        start = offset;
                        return Some(token);
                    }
                }
                let token = &part[start..];
                start = part.len();
                Some(token)
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_tokens_borrow_source_slices_including_repeated_code_parts() {
        let input =
            "rust rust 123 caf\u{e9} \u{3bf}\u{3c2} \u{10428} \u{4e2d} abc_abc _ __the_and__";
        let expected = [
            "rust",
            "rust",
            "123",
            "caf\u{e9}",
            "\u{3bf}\u{3c2}",
            "\u{10428}",
            "\u{4e2d}",
            "abc_abc",
            "abc",
            "_",
            "__the_and__",
        ];
        let actual = tokens(input).collect::<Vec<_>>();
        assert_eq!(actual, expected);
        for token in &actual {
            let Cow::Borrowed(slice) = token else {
                panic!("unchanged token was copied: {token}");
            };
            let offset = slice.as_ptr() as usize - input.as_ptr() as usize;
            assert_eq!(&input[offset..offset + slice.len()], *slice);
        }
        assert_eq!(actual[1].as_ptr() as usize - actual[0].as_ptr() as usize, 5);
        assert_eq!(actual[8].as_ptr(), actual[7].as_ptr());

        let input = "rust 42 caf\u{e9} \u{4e2d} ".repeat(25_000);
        let mut count = 0;
        for token in tokens(&input) {
            assert!(matches!(token, Cow::Borrowed(_)));
            count += 1;
        }
        assert_eq!(count, 100_000);
    }

    #[test]
    fn owned_lowercase_preserves_context_expansion_and_first_occurrence_dedup() {
        for (input, expected) in [
            ("\u{39f}\u{3a3}", vec!["\u{3bf}\u{3c2}"]),
            (
                "\u{39f}\u{3a3}Value",
                vec!["\u{3bf}\u{3c3}value", "\u{3bf}\u{3c2}", "value"],
            ),
            ("\u{130}", vec!["i\u{307}"]),
            ("\u{10400}", vec!["\u{10428}"]),
            ("\u{1c5}", vec!["\u{1c6}"]),
            ("\u{212a}", vec!["k"]),
            ("HTTPServer", vec!["httpserver", "http", "server"]),
            ("Project", vec!["project"]),
        ] {
            let actual = tokens(input).collect::<Vec<_>>();
            assert_eq!(actual, expected, "{input}");
            assert_eq!(actual, token_reference(input), "{input}");
            assert!(actual.iter().all(|token| matches!(token, Cow::Owned(_))));
        }
        let actual = tokens("fooFoo Foo_foo").collect::<Vec<_>>();
        assert_eq!(actual, ["foofoo", "foo", "foo_foo", "foo"]);
        assert!(matches!(actual[1], Cow::Borrowed(_)));
        assert!(matches!(actual[3], Cow::Owned(_)));
    }

    pub(super) fn token_reference(input: &str) -> Vec<String> {
        fn push(tokens: &mut Vec<String>, token: &str) {
            let token = token.to_lowercase();
            if !token.is_empty()
                && !STOP_WORDS.contains(&token.as_str())
                && !tokens.contains(&token)
            {
                tokens.push(token);
            }
        }
        let mut tokens = Vec::new();
        for symbol in input.split(|ch: char| !ch.is_alphanumeric() && ch != '_') {
            if symbol.is_empty() {
                continue;
            }
            let mut symbol_tokens = Vec::new();
            push(&mut symbol_tokens, symbol);
            for part in symbol.split('_').filter(|part| !part.is_empty()) {
                let chars = part.chars().collect::<Vec<_>>();
                let mut start = 0;
                for i in 1..chars.len() {
                    let boundary = (chars[i - 1].is_lowercase() && chars[i].is_uppercase())
                        || (chars[i - 1].is_uppercase()
                            && chars[i].is_uppercase()
                            && chars.get(i + 1).is_some_and(|ch| ch.is_lowercase()));
                    if boundary {
                        push(
                            &mut symbol_tokens,
                            &chars[start..i].iter().collect::<String>(),
                        );
                        start = i;
                    }
                }
                push(
                    &mut symbol_tokens,
                    &chars[start..].iter().collect::<String>(),
                );
            }
            tokens.extend(symbol_tokens);
        }
        tokens
    }

    #[test]
    fn query_unique_terms_match_frozen_symbol_semantics_across_100_sequences() {
        use std::collections::BTreeSet;

        let cases = [
            "",
            "THE and with",
            "fooFoo Foo_foo",
            "_foo_foo_",
            "HTTPServer",
            "Project",
            "src/my_file.rs",
            "42 HTTP2Server",
            "_ __the_and__",
            "CAF\u{c9}",
            "\u{39f}\u{3a3}",
            "\u{39f}\u{3a3}Value",
            "\u{130}",
            "\u{10400}",
            "\u{1c5}",
            "\u{212a}",
            "中文检索",
            "rust rust",
        ];
        for seed in 0..100 {
            let mut input = String::new();
            for position in 0..=seed {
                input.push_str(cases[(position * 7 + seed) % cases.len()]);
                input.push_str([" ", "\n", "/", ".", "\u{2014}"][position % 5]);
            }
            let expected = token_reference(&input).into_iter().collect::<BTreeSet<_>>();
            let actual = query_tokens(&input)
                .map(Cow::into_owned)
                .collect::<BTreeSet<_>>();
            assert_eq!(actual, expected, "seed {seed}");
            assert_eq!(
                query_tokens(&input).collect::<BTreeSet<_>>(),
                tokens(&input).collect::<BTreeSet<_>>(),
                "seed {seed}"
            );
        }
        assert_eq!(query_tokens("fooFoo").count(), 3);
        assert_eq!(tokens("fooFoo").count(), 2);
        assert_eq!(crate::model::token_count_hint("fooFoo fooFoo"), 4);
        assert_eq!(
            query_tokens("\u{39f}\u{3a3}Value").collect::<Vec<_>>(),
            ["\u{3bf}\u{3c3}value", "\u{3bf}\u{3c2}", "value"]
        );
    }

    #[test]
    fn query_token_iteration_does_not_stage_code_symbol_vectors() {
        for count in [1, 7, 257, 4096] {
            let input = (0..count)
                .map(|part| format!("part{part:04}"))
                .collect::<Vec<_>>()
                .join("_");
            let (actual_count, allocations) = crate::test_alloc::allocations(|| {
                query_tokens(&input)
                    .map(|token| {
                        assert!(matches!(token, Cow::Borrowed(_)));
                        std::hint::black_box(token);
                        1
                    })
                    .sum::<usize>()
            });
            assert_eq!(actual_count, if count == 1 { 1 } else { count + 1 });
            assert_eq!(allocations, 0, "parts {count}");
            let (old_count, old_allocations) =
                crate::test_alloc::allocations(|| tokens(&input).map(std::hint::black_box).count());
            assert_eq!(old_count, actual_count);
            assert_eq!(old_allocations == 0, count == 1, "parts {count}");
        }
        for input in ["Project", "\u{39f}\u{3a3}", "\u{130}", "\u{10400}"] {
            let (count, allocations) = crate::test_alloc::allocations(|| {
                query_tokens(input).map(std::hint::black_box).count()
            });
            assert_eq!(count, 1, "{input}");
            let (old_count, old_allocations) =
                crate::test_alloc::allocations(|| tokens(input).map(std::hint::black_box).count());
            assert_eq!(old_count, count, "{input}");
            assert_eq!(allocations, old_allocations, "{input}");
            if input.is_ascii() {
                assert_eq!(allocations, 1, "{input}");
            }
        }
    }

    #[test]
    fn utf8_symbol_slices_match_frozen_character_algorithm_across_100_seeds() {
        let alphabet = [
            'a',
            'Z',
            'b',
            'H',
            '2',
            '_',
            '/',
            '.',
            '\u{130}',
            '\u{df}',
            '\u{3a3}',
            '\u{3c3}',
            '\u{c9}',
            '\u{e9}',
            '\u{307}',
            '\u{10400}',
            '\u{10428}',
            '\u{4e2d}',
        ];
        for seed in 0..100_u64 {
            let mut state = seed + 1;
            let mut input = String::from("HTTPServer __the_and__ fooFoo \u{39f}\u{3a3}Value ");
            for _ in 0..(seed as usize * 31 + 1) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                input.push(alphabet[(state >> 32) as usize % alphabet.len()]);
            }
            let expected = token_reference(&input);
            assert_eq!(tokenize(&input), expected, "seed {seed}");
            assert_eq!(tokens(&input).count(), expected.len(), "seed {seed}");
        }
        for input in [
            "a".repeat(100_000),
            "\u{10400}".repeat(25_000),
            "aB_".repeat(10_000),
        ] {
            assert_eq!(tokenize(&input), token_reference(&input));
        }
    }

    #[test]
    fn incremental_tokens_preserve_frozen_outputs_and_counts_across_100_sequences() {
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            ("THE and with", &[]),
            ("fooFoo", &["foofoo", "foo"]),
            ("_foo_foo_", &["_foo_foo_", "foo"]),
            ("HTTPServer", &["httpserver", "http", "server"]),
            ("src/my_file.rs", &["src", "my_file", "my", "file", "rs"]),
            ("42 HTTP2Server", &["42", "http2server"]),
            ("_", &["_"]),
            ("CAF\u{c9}", &["caf\u{e9}"]),
            ("\u{130}", &["i\u{307}"]),
            ("中文检索", &["中文检索"]),
        ];
        for seed in 0..100 {
            let mut input = String::new();
            let mut expected = Vec::new();
            for position in 0..(seed + 1) {
                let (symbol, output) = cases[(position * 7 + seed) % cases.len()];
                input.push_str(symbol);
                input.push_str([" ", "\n", "/", ".", "\u{2014}"][position % 5]);
                expected.extend(output.iter().map(|token| (*token).to_owned()));
            }
            assert_eq!(tokens(&input).collect::<Vec<_>>(), expected, "seed {seed}");
            assert_eq!(tokenize(&input), expected, "seed {seed}");
            assert_eq!(tokens(&input).count(), expected.len(), "seed {seed}");
            assert_eq!(crate::model::token_count_hint(&input), expected.len());
        }
    }

    #[test]
    fn identity_is_complete_and_immutable() {
        let config = TokenizerConfig::default();
        assert_eq!(
            TokenizerConfig::from_json(&config.to_json()).unwrap(),
            config
        );
        for field in ["name", "version", "profile", "language_hint", "config"] {
            let mut bad = config.to_json();
            bad.as_object_mut().unwrap().remove(field);
            assert!(TokenizerConfig::from_json(&bad).is_err());
        }
        let mut bad = config.to_json();
        bad.as_object_mut()
            .unwrap()
            .insert("extra".into(), JsonValue::Null);
        assert!(TokenizerConfig::from_json(&bad).is_err());
        assert!(TokenizerConfig::from_json(&JsonValue::Null).is_err());
        assert!(TokenizerConfig::from_name("default_en_cjk_bigram").is_err());
    }

    #[test]
    fn symbols_paths_unicode_numbers_and_fixed_stop_words() {
        assert_eq!(
            tokenize("the FooBar foo_bar HTTPServer src/my_file.rs 42 CAFÉ"),
            [
                "foobar",
                "foo",
                "bar",
                "foo_bar",
                "foo",
                "bar",
                "httpserver",
                "http",
                "server",
                "src",
                "my_file",
                "my",
                "file",
                "rs",
                "42",
                "café"
            ]
        );
        assert!(tokenize("THE and with").is_empty());
        assert_eq!(tokenize("foo foo"), ["foo", "foo"]);
        assert_eq!(tokenize("中文检索"), ["中文检索"]);
    }
}
