use std::time::{SystemTime, UNIX_EPOCH};

// u64 milliseconds reach at most a nine-digit year, plus the fractional suffix.
pub(crate) const MAX_TIMESTAMP_BYTES: usize = 29;

pub fn now_rfc3339() -> String {
    format_unix_millis(unix_millis_now())
}

pub(crate) fn unix_millis_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

pub(crate) fn parse_rfc3339_millis(value: &str) -> Option<u64> {
    let seconds = u64::try_from(parse_rfc3339_seconds(value)?).ok()?;
    let fraction = match value.get(19..)? {
        "Z" => 0,
        suffix if suffix.starts_with('.') && suffix.ends_with('Z') => {
            let digits = &suffix[1..suffix.len() - 1];
            if digits.is_empty() || digits.len() > 3 || !digits.bytes().all(|b| b.is_ascii_digit())
            {
                return None;
            }
            digits.parse::<u64>().ok()? * 10u64.pow(3 - digits.len() as u32)
        }
        _ => return None,
    };
    seconds.checked_mul(1000)?.checked_add(fraction)
}

pub(crate) fn format_unix_millis(millis: u64) -> String {
    let mut output = String::with_capacity(24);
    write_unix_millis(&mut output, millis).expect("writing to String is infallible");
    output
}

pub(crate) fn write_unix_millis(
    output: &mut (impl std::fmt::Write + ?Sized),
    millis: u64,
) -> std::fmt::Result {
    write_timestamp(output, (millis / 1000) as i64, (millis % 1000) as u16)
}

pub fn parse_rfc3339_seconds(value: &str) -> Option<i64> {
    if value.len() < 20 {
        return None;
    }
    let year = value.get(0..4)?.parse::<i32>().ok()?;
    let month = value.get(5..7)?.parse::<u32>().ok()?;
    let day = value.get(8..10)?.parse::<u32>().ok()?;
    let hour = value.get(11..13)?.parse::<u32>().ok()?;
    let minute = value.get(14..16)?.parse::<u32>().ok()?;
    let second = value.get(17..19)?.parse::<u32>().ok()?;
    if value.as_bytes().get(4) != Some(&b'-')
        || value.as_bytes().get(7) != Some(&b'-')
        || value.as_bytes().get(10) != Some(&b'T')
        || value.as_bytes().get(13) != Some(&b':')
        || value.as_bytes().get(16) != Some(&b':')
    {
        return None;
    }
    let days_in_month = match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return None,
    };
    if !(1..=days_in_month).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    match value.get(19..)? {
        "Z" => {}
        suffix if suffix.starts_with('.') && suffix.ends_with('Z') => {
            let fraction = &suffix[1..suffix.len() - 1];
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
        }
        _ => return None,
    }
    Some(
        days_from_civil(year, month, day) * 86_400
            + hour as i64 * 3600
            + minute as i64 * 60
            + second as i64,
    )
}

fn write_timestamp(
    output: &mut (impl std::fmt::Write + ?Sized),
    seconds: i64,
    millis: u16,
) -> std::fmt::Result {
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;
    write!(
        output,
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
    )?;
    if millis != 0 {
        write!(output, ".{millis:03}")?;
    }
    output.write_char('Z')
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - (month <= 2) as i32;
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let yoe = year - era * 400;
    let mp = month as i32 + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as i64
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + (m <= 2) as i64;
    (year as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_timestamp_buffer_covers_full_millisecond_range_without_growth() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1, ReservedString};
        for millis in [0, 999, 1000, 253_402_300_799_999, u64::MAX]
            .into_iter()
            .chain((0..100u64).map(|seed| seed.wrapping_mul(6364136223846793005)))
        {
            let expected = format_unix_millis(millis);
            assert!(expected.len() <= MAX_TIMESTAMP_BYTES);
            let context = JsonOperationContext::standalone();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut held = context.workspace_reservation();
            held.grow_to(
                limit - MAX_TIMESTAMP_BYTES,
                limit - MAX_TIMESTAMP_BYTES,
                "operation_scratch",
            )
            .unwrap();
            let mut value =
                ReservedString::with_capacity(MAX_TIMESTAMP_BYTES, &context, "batch_prepare")
                    .unwrap();
            let pointer = value.as_ref().as_ptr();
            value.append(|out| write_unix_millis(out, millis)).unwrap();
            assert_eq!(value.as_ref(), expected);
            assert_eq!(value.as_ref().as_ptr(), pointer);
            let (value, work) = value.into_parts();
            assert_eq!(value.capacity(), MAX_TIMESTAMP_BYTES);
            assert_eq!(work.reserved_bytes(), MAX_TIMESTAMP_BYTES);
            drop(value);
            drop(work);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn streaming_timestamps_preserve_format_and_exact_capacity() {
        let mut values = vec![0, 1, 999, 1000, 253_402_300_799_000, 253_402_300_799_999];
        for seed in 0..100_u64 {
            values.push(seed.wrapping_mul(6364136223846793005) % 253_402_300_800_000);
        }
        for millis in values {
            let seconds = millis / 1000;
            let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
            let base = format!(
                "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
                seconds % 86_400 / 3600,
                seconds % 3600 / 60,
                seconds % 60
            );
            let expected = if millis % 1000 == 0 {
                format!("{base}Z")
            } else {
                format!("{base}.{:03}Z", millis % 1000)
            };
            let mut output = String::with_capacity(expected.len());
            let pointer = output.as_ptr();
            write_unix_millis(&mut output, millis).unwrap();
            assert_eq!(output, expected);
            assert_eq!(output.as_ptr(), pointer);
            assert_eq!(output.capacity(), expected.len());
            assert_eq!(format_unix_millis(millis), expected);
            assert_eq!(parse_rfc3339_millis(&output), Some(millis));
        }
    }

    #[test]
    fn rfc3339_epoch_round_trip() {
        assert_eq!(parse_rfc3339_seconds("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(format_unix_millis(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn milliseconds_round_trip_without_loss() {
        for millis in [0, 1, 999, 1000, 1_791_234_567_890] {
            assert_eq!(
                parse_rfc3339_millis(&format_unix_millis(millis)),
                Some(millis)
            );
        }
        assert_eq!(parse_rfc3339_millis("1970-01-01T00:00:00.1Z"), Some(100));
        for invalid in [
            "1970-01-01T00:00:00Zjunk",
            "1970-01-01T00:00:00.1234Z",
            "1969-01-01T00:00:00Z",
        ] {
            assert_eq!(parse_rfc3339_millis(invalid), None);
        }
    }

    #[test]
    fn utc_timestamps_validate_calendar_and_entire_suffix() {
        for value in [
            "2024-02-29T23:59:59Z",
            "2000-02-29T00:00:00Z",
            "2026-04-30T00:00:00.123Z",
        ] {
            assert!(parse_rfc3339_millis(value).is_some(), "{value}");
        }
        for value in [
            "2023-02-29T00:00:00Z",
            "1900-02-29T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-02-30T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:60:00Z",
            "2026-01-01T00:00:60Z",
            "2026-01-01T00:00:00Zjunk",
            "2026-01-01T00:00:00.Z",
            "2026-01-01T00:00:00.abcZ",
            "2026-01-01 00:00:00Z",
            "2026-01-01T00:00:00+bad",
        ] {
            assert_eq!(parse_rfc3339_seconds(value), None, "{value}");
            assert_eq!(parse_rfc3339_millis(value), None, "{value}");
        }
        for day in 1..=366 {
            let millis = (1_704_067_200 + (day - 1) * 86_400) * 1000;
            assert_eq!(
                parse_rfc3339_millis(&format_unix_millis(millis)),
                Some(millis)
            );
        }
    }
}
