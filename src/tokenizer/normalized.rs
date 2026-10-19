use super::STOP_WORDS;
use crate::json::{JsonOperationContext, ReservedString, WorkspaceReservation};
use crate::Result;
use std::borrow::Cow;

pub(super) struct Token<'a> {
    pub(super) value: Cow<'a, str>,
    pub(super) work: WorkspaceReservation,
}

pub(super) fn normalize<'a>(
    raw: &'a str,
    context: &JsonOperationContext,
) -> Result<Option<Token<'a>>> {
    if raw.is_ascii() {
        if STOP_WORDS.iter().any(|stop| stop.eq_ignore_ascii_case(raw)) {
            return Ok(None);
        }
        if raw.bytes().any(|byte| byte.is_ascii_uppercase()) {
            let (value, work) =
                ReservedString::build_exact(raw.len(), context, "operation_scratch", |out| {
                    for byte in raw.bytes() {
                        out.write_char(byte.to_ascii_lowercase() as char)?;
                    }
                    Ok(())
                })?
                .into_parts();
            return Ok(Some(Token {
                value: Cow::Owned(value),
                work,
            }));
        }
        return Ok(Some(Token {
            value: Cow::Borrowed(raw),
            work: context.workspace_reservation(),
        }));
    }
    let mut work = context.workspace_reservation();
    let value = if raw
        .chars()
        .all(|ch| ch.to_lowercase().eq(std::iter::once(ch)))
    {
        Cow::Borrowed(raw)
    } else {
        let (capacity, peak) = lowercase_layout(raw);
        work.grow_to_wide(peak, peak, "operation_scratch")?;
        let value = raw.to_lowercase();
        if value.capacity() as u128 > capacity {
            return Err(std::io::Error::other(
                "Unicode normalization allocation exceeds verified capacity",
            )
            .into());
        }
        work.retain(value.capacity());
        Cow::Owned(value)
    };
    if STOP_WORDS.contains(&value.as_ref()) {
        return Ok(None);
    }
    Ok(Some(Token { value, work }))
}

// Verified std String growth envelope; toolchain changes require revalidation.
// Contextual sigma changes content, not UTF-8 length. Std still creates content.
pub(super) fn lowercase_layout(raw: &str) -> (u128, u128) {
    let mut capacity = raw.len() as u128;
    let mut peak = capacity;
    let mut length = 0u128;
    for ch in raw.chars().flat_map(char::to_lowercase) {
        length += ch.len_utf8() as u128;
        if length > capacity {
            let next = (2 * capacity).max(length).max(8);
            peak = peak.max(capacity + next);
            capacity = next;
        }
    }
    (capacity, peak)
}
