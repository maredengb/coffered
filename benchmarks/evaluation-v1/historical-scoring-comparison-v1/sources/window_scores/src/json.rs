use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Cursor, Read};

use crate::error::{Error, ErrorCode, Result};

mod build;
mod copy;
mod lines;
mod output;
mod policy;
mod work;
pub(crate) use build::JsonBuilder;
pub(crate) use lines::JsonLines;
use output::JsonWrite;
pub(crate) use output::{
    write_validated_jsonl_fields, write_vector_json, JsonEncodingWorkspace, JsonFieldValue,
    JsonOutputSchema, JsonTextWriter,
};
pub(crate) use policy::{
    is_limit_error, is_memory_budget_error, retry_after_memory_reclaim, JsonInputKind,
    JsonLimitReason, JsonProcessingPolicyV1, JsonSchema,
};
pub(crate) use work::{
    btree_workspace, copy_precharged_string, reserve_heap_slot, try_shrink_heap, CacheReservation,
    CatalogReservation, JsonOperationContext, JsonReservation, ReservedString, ReservedVec,
    WorkLedger, WorkspaceReservation,
};

pub type JsonObject = BTreeMap<String, JsonValue>;

pub(crate) const MAX_JSON_DEPTH: usize = JsonProcessingPolicyV1::MAX_CONTAINER_DEPTH;

#[derive(Default)]
pub enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(JsonObject),
}

impl std::fmt::Debug for JsonValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.validate_tree(Some(MAX_JSON_DEPTH), 0).is_err() {
            return formatter.write_str("JsonValue(<invalid or over-depth>)");
        }
        match self {
            Self::Null => formatter.write_str("Null"),
            Self::Bool(value) => formatter.debug_tuple("Bool").field(value).finish(),
            Self::Number(value) => formatter.debug_tuple("Number").field(value).finish(),
            Self::String(value) => formatter.debug_tuple("String").field(value).finish(),
            Self::Array(value) => formatter.debug_tuple("Array").field(value).finish(),
            Self::Object(value) => formatter.debug_tuple("Object").field(value).finish(),
        }
    }
}

impl Drop for JsonValue {
    fn drop(&mut self) {
        if !matches!(self, Self::Array(values) if !values.is_empty())
            && !matches!(self, Self::Object(values) if !values.is_empty())
        {
            return;
        }
        let mut value = std::mem::take(self);
        let mut pending = Self::Null;
        loop {
            // Each suspended container holds the previous frame in its last
            // child slot. Remove that link on return, so every child is visited once.
            let child = match &mut value {
                Self::Array(values) => values
                    .last_mut()
                    .map(|child| std::mem::replace(child, std::mem::take(&mut pending))),
                Self::Object(values) => values.last_entry().map(|mut entry| {
                    std::mem::replace(entry.get_mut(), std::mem::take(&mut pending))
                }),
                _ => None,
            };
            if let Some(child) = child {
                pending = std::mem::replace(&mut value, child);
            } else if pending.is_null() {
                break;
            } else {
                value = std::mem::take(&mut pending);
                pending = match &mut value {
                    Self::Array(values) => values.pop().unwrap(),
                    Self::Object(values) => values.pop_last().unwrap().1,
                    _ => unreachable!(),
                };
            }
        }
    }
}

impl Clone for JsonValue {
    fn clone(&self) -> Self {
        match self {
            Self::Null => return Self::Null,
            Self::Bool(value) => return Self::Bool(*value),
            Self::Number(value) => return Self::Number(*value),
            Self::String(value) => return Self::String(value.clone()),
            Self::Array(values) if values.is_empty() => return Self::Array(Vec::new()),
            Self::Object(values) if values.is_empty() => return Self::object(),
            _ => {}
        }
        enum Task<'a> {
            Value(&'a JsonValue),
            Array(usize),
            Object(&'a JsonObject),
        }
        let mut tasks = vec![Task::Value(self)];
        let mut values = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                Task::Value(value) => match value {
                    JsonValue::Null => values.push(JsonValue::Null),
                    JsonValue::Bool(value) => values.push(JsonValue::Bool(*value)),
                    JsonValue::Number(value) => values.push(JsonValue::Number(*value)),
                    JsonValue::String(value) => values.push(JsonValue::String(value.clone())),
                    JsonValue::Array(items) => {
                        tasks.push(Task::Array(items.len()));
                        tasks.extend(items.iter().rev().map(Task::Value));
                    }
                    JsonValue::Object(items) => {
                        tasks.push(Task::Object(items));
                        tasks.extend(items.values().rev().map(Task::Value));
                    }
                },
                Task::Array(len) => {
                    let items = values.split_off(values.len() - len);
                    values.push(JsonValue::Array(items));
                }
                Task::Object(object) => {
                    let items = values.split_off(values.len() - object.len());
                    values.push(JsonValue::Object(
                        object.keys().cloned().zip(items).collect(),
                    ));
                }
            }
        }
        values.pop().unwrap()
    }
}

impl PartialEq for JsonValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => return true,
            (Self::Bool(left), Self::Bool(right)) => return left == right,
            (Self::Number(left), Self::Number(right)) => return left == right,
            (Self::String(left), Self::String(right)) => return left == right,
            (Self::Array(_), Self::Array(_)) | (Self::Object(_), Self::Object(_)) => {}
            _ => return false,
        }
        let mut pending = vec![(self, other)];
        while let Some((left, right)) = pending.pop() {
            match (left, right) {
                (JsonValue::Null, JsonValue::Null) => {}
                (JsonValue::Bool(a), JsonValue::Bool(b)) if a == b => {}
                (JsonValue::Number(a), JsonValue::Number(b)) if a == b => {}
                (JsonValue::String(a), JsonValue::String(b)) if a == b => {}
                (JsonValue::Array(a), JsonValue::Array(b)) if a.len() == b.len() => {
                    pending.extend(a.iter().zip(b));
                }
                (JsonValue::Object(a), JsonValue::Object(b)) if a.len() == b.len() => {
                    for ((key_a, value_a), (key_b, value_b)) in a.iter().zip(b) {
                        if key_a != key_b {
                            return false;
                        }
                        pending.push((value_a, value_b));
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

impl JsonValue {
    pub fn object() -> Self {
        JsonValue::Object(BTreeMap::new())
    }

    pub fn as_object(&self) -> Option<&JsonObject> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut JsonObject> {
        match self {
            JsonValue::Object(obj) => Some(obj),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|obj| obj.get(key))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub fn to_compact_string(&self) -> String {
        self.validate_finite_numbers()
            .expect("cannot serialize non-finite JSON number");
        self.compact_string_unchecked()
    }

    pub fn try_to_compact_string(&self) -> Result<String> {
        self.try_to_compact_string_with_kind(JsonInputKind::JsonlRecord)
    }

    pub(crate) fn try_to_compact_string_with_kind(&self, kind: JsonInputKind) -> Result<String> {
        let context = JsonOperationContext::standalone();
        let (mut reservation, bytes) = self.guard_in(kind, &context)?;
        let mut frames = Vec::new();
        reservation.reserve_vec(&mut frames, MAX_JSON_DEPTH + 1, kind)?;
        let mut out = output::JsonTextWriter::new(&context, kind);
        out.reserve(bytes)?;
        let written = write_json_with_frames(
            self,
            &mut out,
            0,
            false,
            &mut frames,
            JsonOutputSchema::Value(kind),
        );
        out.finish(written).map(|(text, _)| text)
    }

    #[cfg(test)]
    fn canonical_digest(&self, kind: JsonInputKind) -> Result<(usize, String)> {
        let context = JsonOperationContext::standalone();
        let (_input, _) = self.guard_in(kind, &context)?;
        self.canonical_digest_validated_in(kind, &context)
            .map(|(bytes, hash, _)| (bytes, hash))
    }

    pub(crate) fn canonical_digest_validated_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(usize, String, JsonReservation)> {
        struct Digest {
            hash: crate::hash::Sha256,
            bytes: usize,
        }
        impl std::fmt::Write for Digest {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                self.hash.update(value.as_bytes());
                self.bytes += value.len();
                Ok(())
            }
        }
        impl JsonWrite for Digest {}
        let mut reservation = context.reservation();
        reservation.grow(std::mem::size_of::<Digest>() + 64, kind)?;
        let mut frames = Vec::new();
        reservation.reserve_vec(&mut frames, MAX_JSON_DEPTH + 1, kind)?;
        let mut digest = Digest {
            hash: crate::hash::Sha256::default(),
            bytes: 0,
        };
        write_json_with_frames(
            self,
            &mut digest,
            0,
            false,
            &mut frames,
            JsonOutputSchema::Value(kind),
        )
        .unwrap();
        let frame_bytes = frames.capacity() * std::mem::size_of::<WriteFrame<'_>>();
        drop(frames);
        let hash = crate::hash::digest_hex(digest.hash.finish());
        reservation.release(frame_bytes + std::mem::size_of::<Digest>());
        Ok((digest.bytes, hash, reservation))
    }

    fn compact_string_unchecked(&self) -> String {
        let mut out = String::new();
        write_json(self, &mut out, 0, false);
        out
    }

    pub fn to_pretty_string(&self) -> String {
        self.validate_finite_numbers()
            .expect("cannot serialize non-finite JSON number");
        let mut out = String::new();
        write_json(self, &mut out, 0, true);
        out
    }

    pub fn validate_finite_numbers(&self) -> Result<()> {
        self.validate_tree(None, 0)
    }

    pub(crate) fn validate_kind(&self, kind: JsonInputKind) -> Result<()> {
        self.guard_in(kind, &JsonOperationContext::standalone())
            .map(|_| ())
    }

    pub(crate) fn guard_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(JsonReservation, usize)> {
        let mut reservation = context.reservation();
        let layout = self.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(&mut reservation),
            true,
        )?;
        Ok((reservation, layout.canonical_bytes))
    }

    pub(crate) fn canonical_bytes_retained_in(
        &self,
        kind: JsonInputKind,
        retained: &mut JsonReservation,
    ) -> Result<usize> {
        self.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(retained),
            true,
        )
        .map(|layout| layout.canonical_bytes)
    }

    fn validate_tree(&self, depth_limit: Option<usize>, enclosing_depth: usize) -> Result<()> {
        self.validate_tree_layout(depth_limit, enclosing_depth, None, None, true)
            .map(|_| ())
    }

    // The source was already validated with its schema; do not turn an envelope into a value root.
    pub(crate) fn reserve_retained_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<JsonReservation> {
        let mut work = context.reservation();
        self.reconcile_retained_in(kind, &mut work)?;
        Ok(work)
    }

    // Reuse the reservation when owned JSON moves between representations. Scratch
    // is charged separately so traversal cannot spend credit backing live nodes.
    pub(crate) fn reconcile_retained_in(
        &self,
        kind: JsonInputKind,
        work: &mut JsonReservation,
    ) -> Result<()> {
        self.validate_tree_layout(
            None,
            0,
            Some(JsonOutputSchema::Value(kind)),
            Some(work),
            false,
        )?;
        Ok(())
    }

    fn validate_tree_layout(
        &self,
        depth_limit: Option<usize>,
        enclosing_depth: usize,
        schema: Option<JsonOutputSchema>,
        mut reservation: Option<&mut JsonReservation>,
        measure_canonical: bool,
    ) -> Result<JsonLayout> {
        let scratch = reservation.as_ref().map(|work| work.scratch());
        self.validate_tree_layout_with_scratch(
            depth_limit,
            enclosing_depth,
            schema,
            reservation.take(),
            scratch,
            measure_canonical,
        )
    }

    pub(crate) fn retained_bytes_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<usize> {
        self.validate_tree_layout_with_scratch(
            None,
            0,
            Some(JsonOutputSchema::Value(kind)),
            None,
            Some(context.reservation()),
            false,
        )
        .map(|layout| layout.retained_bytes)
    }

    // The caller retains source ownership; only validation scratch is new work.
    pub(crate) fn encoding_shape_in(
        &self,
        kind: JsonInputKind,
        context: &JsonOperationContext,
    ) -> Result<(usize, usize)> {
        self.validate_tree_layout_with_scratch(
            Some(MAX_JSON_DEPTH),
            0,
            Some(JsonOutputSchema::Value(kind)),
            None,
            Some(context.reservation()),
            true,
        )
        .map(|layout| (layout.depth + 1, layout.retained_bytes))
    }

    fn validate_tree_layout_with_scratch(
        &self,
        depth_limit: Option<usize>,
        enclosing_depth: usize,
        schema: Option<JsonOutputSchema>,
        mut reservation: Option<&mut JsonReservation>,
        mut scratch: Option<JsonReservation>,
        measure_canonical: bool,
    ) -> Result<JsonLayout> {
        enum Children<'a> {
            Array(std::slice::Iter<'a, JsonValue>),
            Object(std::collections::btree_map::Iter<'a, String, JsonValue>),
        }
        impl<'a> Iterator for Children<'a> {
            type Item = (Option<&'a str>, &'a JsonValue);
            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    Self::Array(values) => values.next().map(|value| (None, value)),
                    Self::Object(values) => values
                        .next()
                        .map(|(key, value)| (Some(key.as_str()), value)),
                }
            }
        }
        let measure_allocations = scratch.is_some();
        let mut stack = Vec::<(Children<'_>, Option<JsonOutputSchema>)>::new();
        let mut retained_bytes = 0usize;
        let mut next = Some((self, schema));
        let mut canonical_bytes = 0usize;
        let mut max_depth = 0;
        while let Some((value, schema)) = next {
            let kind = schema.map(JsonOutputSchema::kind);
            let children = match value {
                JsonValue::Number(value) if !value.is_finite() => {
                    return Err(json_error("JSON numbers must be finite"));
                }
                JsonValue::Array(values) => Some(Children::Array(values.iter())),
                JsonValue::Object(values) => Some(Children::Object(values.iter())),
                _ => None,
            };
            if let Some(children) = children {
                if depth_limit
                    .is_some_and(|limit| stack.len().saturating_add(enclosing_depth) >= limit)
                {
                    return Err(kind.unwrap_or(JsonInputKind::JsonlRecord).limit_error(
                        JsonLimitReason::Depth,
                        MAX_JSON_DEPTH,
                        Some(
                            stack
                                .len()
                                .saturating_add(enclosing_depth)
                                .saturating_add(1),
                        ),
                    ));
                }
                if let Some(work) = &mut scratch {
                    if let Err(error) = work.reserve_vec(&mut stack, 1, kind.unwrap()) {
                        let can_trim = reservation.as_ref().is_some_and(|retained| {
                            retained.reserved_bytes() > retained.covered_bytes()
                        });
                        if !is_limit_error(&error) || !can_trim {
                            return Err(error);
                        }
                        let retained = reservation.as_mut().unwrap();
                        retained.retain(retained.covered_bytes());
                        work.reserve_vec(&mut stack, 1, kind.unwrap())?;
                    }
                }
                stack.push((children, schema));
                max_depth = max_depth.max(stack.len());
            }
            if let Some(kind) = kind {
                if measure_canonical {
                    canonical_bytes =
                        canonical_bytes.saturating_add(shallow_canonical_bytes(value));
                }
                if measure_canonical
                    && canonical_bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES
                {
                    return Err(kind.limit_error(
                        JsonLimitReason::UnitBytes,
                        JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                        Some(canonical_bytes),
                    ));
                }
                if measure_allocations {
                    let bytes = match value {
                        JsonValue::String(value) => value.capacity(),
                        JsonValue::Array(values) => values
                            .capacity()
                            .saturating_mul(std::mem::size_of::<JsonValue>()),
                        JsonValue::Object(values) => values
                            .keys()
                            .fold(work::object_workspace(values.len()), |bytes, key| {
                                bytes.saturating_add(key.capacity())
                            }),
                        _ => 0,
                    };
                    retained_bytes = retained_bytes.saturating_add(bytes);
                    if let Some(work) = &mut reservation {
                        work.cover(retained_bytes, kind)?;
                    }
                }
            }
            next = loop {
                let Some((children, schema)) = stack.last_mut() else {
                    break None;
                };
                if let Some((key, value)) = children.next() {
                    let schema = schema.map(|schema| match key {
                        Some(key) => schema.field(key),
                        None => schema.element(),
                    });
                    break Some((value, schema));
                }
                stack.pop();
            };
        }
        if let Some(work) = &mut reservation {
            work.retain(retained_bytes);
        }
        Ok(JsonLayout {
            canonical_bytes,
            depth: max_depth,
            retained_bytes,
        })
    }
}

struct JsonLayout {
    retained_bytes: usize,
    canonical_bytes: usize,
    depth: usize,
}

// Validate normalized typed objects without cloning their untrusted strings.
pub(crate) struct JsonObjectGuard {
    work: JsonReservation,
    kind: JsonInputKind,
    fields: usize,
    bytes: usize,
}

impl JsonObjectGuard {
    pub(crate) fn new(kind: JsonInputKind, context: &JsonOperationContext) -> Self {
        Self {
            work: context.reservation(),
            kind,
            fields: 0,
            bytes: 2,
        }
    }

    pub(crate) fn string_field(&mut self, key: &str, value: &String) -> Result<()> {
        self.field(key, string_canonical_bytes(value))?;
        self.work.grow(value.capacity(), self.kind)
    }

    pub(crate) fn value_field(&mut self, key: &str, value: &JsonValue) -> Result<()> {
        let mut work = self.work.scratch();
        let layout = value.validate_tree_layout(
            Some(MAX_JSON_DEPTH),
            1,
            Some(JsonOutputSchema::Value(self.kind)),
            Some(&mut work),
            true,
        )?;
        self.field(key, layout.canonical_bytes)?;
        self.work.merge(work);
        Ok(())
    }

    fn field(&mut self, key: &str, value_bytes: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .saturating_add(string_canonical_bytes(key))
            .saturating_add(1 + usize::from(self.fields != 0))
            .saturating_add(value_bytes);
        if self.bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
            return Err(self.kind.limit_error(
                JsonLimitReason::UnitBytes,
                JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                Some(self.bytes),
            ));
        }
        self.work.grow(
            work::object_workspace(self.fields + 1) - work::object_workspace(self.fields)
                + key.len(),
            self.kind,
        )?;
        self.fields += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> JsonReservation {
        self.work
    }
}

pub fn object_from_pairs<I, K>(pairs: I) -> JsonValue
where
    I: IntoIterator<Item = (K, JsonValue)>,
    K: Into<String>,
{
    JsonValue::Object(
        pairs
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

pub fn parse(input: &str) -> Result<JsonValue> {
    parse_with_schema(input, JsonSchema::Value(JsonInputKind::JsonlRecord))
}

#[cfg(test)]
pub(crate) fn parse_with_kind(input: &str, kind: JsonInputKind) -> Result<JsonValue> {
    parse_with_schema(input, JsonSchema::Value(kind))
}

pub(crate) fn parse_with_schema(input: &str, schema: JsonSchema) -> Result<JsonValue> {
    parse_with_schema_in(input, schema, &JsonOperationContext::standalone())
        .map(ProcessedJson::into_value)
}

pub(crate) struct ProcessedJson {
    pub(crate) value: JsonValue,
    _reservation: JsonReservation,
}

impl ProcessedJson {
    fn into_value(self) -> JsonValue {
        self.value
    }

    pub(crate) fn into_parts(self) -> (JsonValue, JsonReservation) {
        (self.value, self._reservation)
    }
}

pub(crate) fn parse_with_schema_in(
    input: &str,
    schema: JsonSchema,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    parse_buffered(Cursor::new(input.as_bytes()), schema, context, false)
}

pub(crate) fn parse_reader(source: impl Read, schema: JsonSchema) -> Result<JsonValue> {
    parse_reader_in(source, schema, &JsonOperationContext::standalone())
        .map(ProcessedJson::into_value)
}

pub(crate) fn parse_reader_in(
    source: impl Read,
    schema: JsonSchema,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    read_json(source, schema, context, false, 8192)
}

#[cfg(test)]
pub(crate) fn parse_canonical_reader(source: impl Read, kind: JsonInputKind) -> Result<JsonValue> {
    parse_canonical_reader_in(
        &mut source.take(u64::MAX),
        kind,
        &JsonOperationContext::standalone(),
    )
    .map(ProcessedJson::into_value)
}

pub(crate) fn parse_canonical_reader_in(
    source: &mut std::io::Take<impl Read>,
    kind: JsonInputKind,
    context: &JsonOperationContext,
) -> Result<ProcessedJson> {
    // Stored fields already have a checked byte frame. Do not allocate an 8-KiB
    // buffer for every small metadata object in a primary-record scan.
    let buffer_bytes = source.limit().min(8192) as usize;
    read_json(source, JsonSchema::Value(kind), context, true, buffer_bytes)
}

fn read_json(
    source: impl Read,
    schema: JsonSchema,
    context: &JsonOperationContext,
    canonical: bool,
    buffer_bytes: usize,
) -> Result<ProcessedJson> {
    let mut input_work = context.reservation();
    input_work.grow(buffer_bytes, schema.kind())?;
    let source = BufReader::with_capacity(buffer_bytes, source);
    parse_buffered(source, schema, context, canonical)
}

fn parse_buffered(
    source: impl BufRead,
    schema: JsonSchema,
    context: &JsonOperationContext,
    canonical: bool,
) -> Result<ProcessedJson> {
    let mut parser = Parser {
        source,
        canonical,
        pos: 0,
        unit: None,
        kind: schema.kind(),
        work: context.aggregate_reservation(),
    };
    let value = parser.parse_value(0, schema)?;
    parser.skip_ws()?;
    if parser.peek()?.is_some() {
        return Err(json_error(format!(
            "unexpected trailing data at byte {}",
            parser.pos
        )));
    }
    parser.work.finish_aggregate();
    Ok(ProcessedJson {
        value,
        _reservation: parser.work,
    })
}

pub fn required_string(obj: &JsonObject, key: &str) -> Result<String> {
    obj.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MissingRequiredField,
                format!("missing string field {key}"),
            )
        })
}

pub(crate) fn take_required_string(obj: &mut JsonObject, key: &str) -> Result<String> {
    match obj.get_mut(key) {
        Some(JsonValue::String(value)) => Ok(std::mem::take(value)),
        _ => Err(Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing string field {key}"),
        )),
    }
}

pub fn optional_string(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

pub fn required_usize(obj: &JsonObject, key: &str) -> Result<usize> {
    let value = obj.get(key).and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing numeric field {key}"),
        )
    })?;
    let exclusive_max = (usize::MAX as u128 + 1) as f64;
    if !value.is_finite() || value < 0.0 || value >= exclusive_max || value.fract() != 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            format!("{key} must be a non-negative integer"),
        ));
    }
    Ok(value as usize)
}

pub fn required_array<'a>(obj: &'a JsonObject, key: &str) -> Result<&'a [JsonValue]> {
    obj.get(key).and_then(JsonValue::as_array).ok_or_else(|| {
        Error::new(
            ErrorCode::MissingRequiredField,
            format!("missing array field {key}"),
        )
    })
}

pub fn ensure_object(value: JsonValue) -> Result<JsonValue> {
    if matches!(value, JsonValue::Object(_)) {
        Ok(value)
    } else {
        Err(Error::new(
            ErrorCode::InvalidJsonlSchema,
            "metadata/config fields must be JSON objects",
        ))
    }
}

fn json_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidJsonlSchema, message)
}

#[cfg(test)]
pub(crate) fn is_depth_error(error: &Error) -> bool {
    error
        .detail()
        .iter()
        .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded")
}

fn string_canonical_bytes(value: &str) -> usize {
    value.chars().fold(2usize, |len, ch| {
        len.saturating_add(char_canonical_bytes(ch))
    })
}

fn char_canonical_bytes(ch: char) -> usize {
    match ch {
        '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}' => 2,
        ch if ch.is_control() => 6,
        ch => ch.len_utf8(),
    }
}

fn shallow_canonical_bytes(value: &JsonValue) -> usize {
    match value {
        JsonValue::Null => 4,
        JsonValue::Bool(value) => {
            if *value {
                4
            } else {
                5
            }
        }
        JsonValue::Number(value) => number_canonical_bytes(*value),
        JsonValue::String(value) => string_canonical_bytes(value),
        JsonValue::Array(values) => 2usize.saturating_add(values.len().saturating_sub(1)),
        JsonValue::Object(values) => values.keys().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |len, key| {
                len.saturating_add(string_canonical_bytes(key))
                    .saturating_add(1)
            },
        ),
    }
}

fn number_canonical_bytes(value: f64) -> usize {
    use std::fmt::Write;
    struct Counter(usize);
    impl Write for Counter {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 += value.len();
            Ok(())
        }
    }
    let mut counter = Counter(0);
    write_number(value, &mut counter).unwrap();
    counter.0
}

fn write_number(value: f64, out: &mut impl std::fmt::Write) -> std::fmt::Result {
    if value == 0.0 {
        out.write_char('0')
    } else if value.fract() == 0.0 {
        write!(out, "{value:.0}")
    } else {
        write!(out, "{value}")
    }
}

fn is_canonical_number(value: f64, raw: &str) -> bool {
    struct Comparison<'a>(&'a str);
    impl std::fmt::Write for Comparison<'_> {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0 = self.0.strip_prefix(text).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut comparison = Comparison(raw);
    write_number(value, &mut comparison).is_ok() && comparison.0.is_empty()
}

fn canonical_error() -> Error {
    Error::new(
        ErrorCode::FileFormatError,
        "stored JSON is not in canonical form",
    )
}

pub(crate) fn validate_query_vector(vector: &[f32]) -> Result<()> {
    let mut bytes = 2usize.saturating_add(vector.len().saturating_sub(1));
    for &value in vector {
        if !value.is_finite() {
            return Err(Error::new(
                ErrorCode::InvalidVectorLength,
                "query embedding vector entries must be finite f32 values",
            ));
        }
        bytes = bytes.saturating_add(shallow_canonical_bytes(&JsonValue::Number(f64::from(
            value,
        ))));
        if bytes > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
            return Err(JsonInputKind::QueryEmbedding.limit_error(
                JsonLimitReason::UnitBytes,
                JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                Some(bytes),
            ));
        }
    }
    Ok(())
}

fn validate_integer_precision(raw: &str) -> Result<()> {
    let unsigned = raw.strip_prefix('-').unwrap_or(raw);
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => {
            let exponent = &unsigned[index + 1..];
            let value = exponent.parse::<i64>().unwrap_or_else(|_| {
                if exponent.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                }
            });
            (&unsigned[..index], value)
        }
        None => (unsigned, 0),
    };
    let fractional_digits = mantissa
        .find('.')
        .map(|index| mantissa.len() - index - 1)
        .unwrap_or(0);
    let mut first_nonzero = None;
    let mut last_nonzero = 0;
    let mut digit_count = 0usize;
    for byte in mantissa.bytes().filter(|byte| *byte != b'.') {
        if byte != b'0' {
            first_nonzero.get_or_insert(digit_count);
            last_nonzero = digit_count;
        }
        digit_count += 1;
    }
    let Some(first_nonzero) = first_nonzero else {
        return Ok(());
    };
    let zeros = exponent
        .saturating_sub(fractional_digits as i64)
        .saturating_add((digit_count - last_nonzero - 1) as i64);
    if zeros < 0 {
        return Ok(());
    }
    let precision_error =
        || json_error("integer cannot be represented exactly as binary64; encode it as a string");
    let zeros = usize::try_from(zeros).map_err(|_| precision_error())?;
    let significant_digits = last_nonzero - first_nonzero + 1;
    let len = significant_digits.saturating_add(zeros);
    if len > 309 {
        return Err(precision_error());
    }
    if len <= 15 {
        return Ok(());
    }
    let mut storage = [0u8; 309];
    for (slot, byte) in storage.iter_mut().zip(
        mantissa
            .bytes()
            .filter(|byte| *byte != b'.')
            .skip(first_nonzero)
            .take(significant_digits),
    ) {
        *slot = byte - b'0';
    }
    let digits = &mut storage[..len];
    if let Some(value) = digits.iter().try_fold(0u64, |value, digit| {
        value.checked_mul(10)?.checked_add(u64::from(*digit))
    }) {
        return if 64 - value.leading_zeros() - value.trailing_zeros() <= 53 {
            Ok(())
        } else {
            Err(precision_error())
        };
    }

    // Divide the bounded decimal integer before any float conversion. Binary64
    // represents it exactly only if at most 53 bits remain after trailing zeros.
    let mut bits = 0;
    let mut trailing_zero_bits = 0;
    let mut seen_one = false;
    while digits.iter().any(|digit| *digit != 0) {
        let mut remainder = 0;
        for digit in digits.iter_mut() {
            let value = remainder * 10 + *digit;
            *digit = value / 2;
            remainder = value % 2;
        }
        bits += 1;
        if !seen_one && remainder == 0 {
            trailing_zero_bits += 1;
        }
        seen_one |= remainder != 0;
    }
    if bits <= 1024 && bits - trailing_zero_bits <= 53 {
        Ok(())
    } else {
        Err(precision_error())
    }
}

enum WriteFrame<'a> {
    Value(&'a JsonValue, usize, JsonOutputSchema),
    Array(
        std::slice::Iter<'a, JsonValue>,
        usize,
        bool,
        JsonOutputSchema,
    ),
    Object(
        std::collections::btree_map::Iter<'a, String, JsonValue>,
        usize,
        bool,
        JsonOutputSchema,
    ),
}

#[cfg(test)]
pub(crate) fn test_write_frame_bytes() -> usize {
    std::mem::size_of::<WriteFrame<'_>>()
}

fn write_json(value: &JsonValue, out: &mut String, indent: usize, pretty: bool) {
    write_json_with_frames(
        value,
        out,
        indent,
        pretty,
        &mut Vec::new(),
        JsonOutputSchema::Value(JsonInputKind::JsonlRecord),
    )
    .unwrap();
}

fn write_json_with_frames<'a, W: JsonWrite>(
    value: &'a JsonValue,
    out: &mut W,
    indent: usize,
    pretty: bool,
    frames: &mut Vec<WriteFrame<'a>>,
    schema: JsonOutputSchema,
) -> std::fmt::Result {
    write_json_frames(value, out, indent, pretty, frames, schema, false)
}

fn write_json_frames<'a, W: JsonWrite>(
    value: &'a JsonValue,
    out: &mut W,
    indent: usize,
    pretty: bool,
    frames: &mut Vec<WriteFrame<'a>>,
    schema: JsonOutputSchema,
    bounded: bool,
) -> std::fmt::Result {
    use WriteFrame as Frame;
    let newline = |out: &mut W, indent: usize| -> std::fmt::Result {
        if pretty {
            out.write_char('\n')?;
            for _ in 0..indent {
                out.write_char(' ')?;
            }
        }
        Ok(())
    };
    if bounded && frames.capacity() == 0 {
        return Err(std::fmt::Error);
    }
    frames.push(Frame::Value(value, indent, schema));
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Value(value, indent, schema) => {
                out.set_kind(schema.kind());
                match value {
                    JsonValue::Null => out.write_str("null")?,
                    JsonValue::Bool(value) => {
                        out.write_str(if *value { "true" } else { "false" })?
                    }
                    JsonValue::Number(value) => write_number(*value, out)?,
                    JsonValue::String(value) => write_string(value, out)?,
                    JsonValue::Array(values) => {
                        out.write_char('[')?;
                        frames.push(Frame::Array(values.iter(), indent, true, schema));
                    }
                    JsonValue::Object(values) => {
                        out.write_char('{')?;
                        frames.push(Frame::Object(values.iter(), indent, true, schema));
                    }
                }
            }
            Frame::Array(mut items, indent, first, schema) => {
                out.set_kind(schema.kind());
                if let Some(item) = items.next() {
                    if bounded && frames.capacity() - frames.len() < 2 {
                        return Err(std::fmt::Error);
                    }
                    if !first {
                        out.write_char(',')?;
                    }
                    newline(out, indent + 2)?;
                    frames.push(Frame::Array(items, indent, false, schema));
                    frames.push(Frame::Value(item, indent + 2, schema.element()));
                } else {
                    if !first {
                        newline(out, indent)?;
                    }
                    out.write_char(']')?;
                }
            }
            Frame::Object(mut items, indent, first, schema) => {
                out.set_kind(schema.kind());
                if let Some((key, item)) = items.next() {
                    if bounded && frames.capacity() - frames.len() < 2 {
                        return Err(std::fmt::Error);
                    }
                    if !first {
                        out.write_char(',')?;
                    }
                    newline(out, indent + 2)?;
                    write_string(key, out)?;
                    out.write_char(':')?;
                    if pretty {
                        out.write_char(' ')?;
                    }
                    frames.push(Frame::Object(items, indent, false, schema));
                    frames.push(Frame::Value(item, indent + 2, schema.field(key)));
                } else {
                    if !first {
                        newline(out, indent)?;
                    }
                    out.write_char('}')?;
                }
            }
        }
    }
    Ok(())
}

fn write_string(value: &str, out: &mut impl std::fmt::Write) -> std::fmt::Result {
    out.write_char('"')?;
    let mut start = 0;
    for (offset, ch) in value.char_indices() {
        if ch != '"' && ch != '\\' && !ch.is_control() {
            continue;
        }
        out.write_str(&value[start..offset])?;
        match ch {
            '"' => out.write_str("\\\"")?,
            '\\' => out.write_str("\\\\")?,
            '\n' => out.write_str("\\n")?,
            '\r' => out.write_str("\\r")?,
            '\t' => out.write_str("\\t")?,
            '\u{08}' => out.write_str("\\b")?,
            '\u{0c}' => out.write_str("\\f")?,
            ch => write!(out, "\\u{:04x}", ch as u32)?,
        }
        start = offset + ch.len_utf8();
    }
    out.write_str(&value[start..])?;
    out.write_char('"')
}

struct Parser<R> {
    source: R,
    canonical: bool,
    pos: usize,
    unit: Option<(JsonInputKind, usize)>,
    kind: JsonInputKind,
    work: JsonReservation,
}

impl<R: BufRead> Parser<R> {
    fn parse_value(&mut self, depth: usize, schema: JsonSchema) -> Result<JsonValue> {
        struct Frame {
            value: JsonValue,
            key: Option<String>,
            schema: JsonSchema,
            depth: usize,
        }
        let mut frames = Vec::<Frame>::new();
        let mut next = Some((schema, depth));
        let mut ready = None;
        loop {
            if let Some((schema, depth)) = next.take() {
                self.kind = schema.kind();
                self.skip_ws()?;
                if depth == 0 {
                    self.unit = match schema {
                        JsonSchema::Value(kind) => Some((kind, 0)),
                        _ => None,
                    };
                }
                if schema.is_record() && self.peek()? != Some(b'{') {
                    return Err(json_error("JSON record must be an object"));
                }
                if matches!(self.peek()?, Some(b'[' | b'{')) {
                    if depth >= MAX_JSON_DEPTH {
                        return Err(schema
                            .kind()
                            .limit_error(JsonLimitReason::Depth, MAX_JSON_DEPTH, Some(depth + 1))
                            .with_detail("byte_offset", self.pos));
                    }
                    self.charge_unit(2)?;
                    let value = if self.next()? == Some(b'[') {
                        JsonValue::Array(Vec::new())
                    } else {
                        JsonValue::object()
                    };
                    self.work.reserve_vec(&mut frames, 1, self.kind)?;
                    frames.push(Frame {
                        value,
                        key: None,
                        schema,
                        depth: depth + 1,
                    });
                } else {
                    let value = match self.peek()? {
                        Some(b'n') => {
                            self.expect_literal(b"null")?;
                            JsonValue::Null
                        }
                        Some(b't') => {
                            self.expect_literal(b"true")?;
                            JsonValue::Bool(true)
                        }
                        Some(b'f') => {
                            self.expect_literal(b"false")?;
                            JsonValue::Bool(false)
                        }
                        Some(b'"') => JsonValue::String(self.parse_string()?),
                        Some(b'-' | b'0'..=b'9') => self.parse_number()?,
                        Some(_) => {
                            return Err(json_error(format!("unexpected byte at {}", self.pos)))
                        }
                        None => return Err(json_error("unexpected end of input")),
                    };
                    if let (Some((JsonInputKind::QueryEmbedding, _)), JsonValue::Number(number)) =
                        (self.unit, &value)
                    {
                        let number = *number as f32;
                        if !number.is_finite() {
                            return Err(Error::new(
                                ErrorCode::InvalidVectorLength,
                                "query embedding vector entries must be finite f32 values",
                            ));
                        }
                        self.charge_unit(shallow_canonical_bytes(&JsonValue::Number(f64::from(
                            number,
                        ))))?;
                    } else if !matches!(value, JsonValue::String(_)) {
                        self.charge_unit(shallow_canonical_bytes(&value))?;
                    }
                    ready = Some(value);
                }
            }
            let Some(frame) = frames.last_mut() else {
                let bytes = frames.capacity() * std::mem::size_of::<Frame>();
                drop(frames);
                self.work.release(bytes);
                return Ok(ready.unwrap());
            };
            let after_value = ready.is_some();
            self.kind = frame.schema.kind();
            if let Some(value) = ready.take() {
                match &mut frame.value {
                    JsonValue::Array(values) => {
                        self.work.reserve_vec(values, 1, self.kind)?;
                        values.push(value);
                    }
                    JsonValue::Object(values) => {
                        self.work.grow(
                            work::object_workspace(values.len() + 1)
                                - work::object_workspace(values.len()),
                            self.kind,
                        )?;
                        values.insert(frame.key.take().unwrap(), value);
                    }
                    _ => unreachable!(),
                }
                if frame.schema.is_record() {
                    self.unit = None;
                }
            }
            self.kind = frame.schema.kind();
            let closing = if matches!(frame.value, JsonValue::Array(_)) {
                b']'
            } else {
                b'}'
            };
            self.skip_ws()?;
            if self.consume_if(closing)? {
                ready = Some(frames.pop().unwrap().value);
                continue;
            }
            if after_value {
                self.expect_byte(b',')?;
                self.charge_unit(1)?;
            }
            if let JsonValue::Object(values) = &frame.value {
                self.skip_ws()?;
                let key = self.parse_string()?;
                self.skip_ws()?;
                self.expect_byte(b':')?;
                self.charge_unit(1)?;
                if values.contains_key(&key) {
                    return Err(json_error("duplicate object key"));
                }
                if self.canonical && values.last_key_value().is_some_and(|(last, _)| last > &key) {
                    return Err(canonical_error());
                }
                next = Some((
                    frame.schema.field(&key),
                    if frame.schema.is_record() {
                        0
                    } else {
                        frame.depth
                    },
                ));
                frame.key = Some(key);
            } else {
                next = Some((frame.schema, frame.depth));
            }
        }
    }

    fn charge_unit(&mut self, bytes: usize) -> Result<()> {
        if let Some((kind, used)) = &mut self.unit {
            *used = used.saturating_add(bytes);
            if *used > JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES {
                return Err(kind.limit_error(
                    JsonLimitReason::UnitBytes,
                    JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES,
                    Some(*used),
                ));
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect_byte(b'"')?;
        self.charge_unit(2)?;
        let mut out = String::new();
        while self.peek()?.is_some() {
            // Most text is unescaped ASCII; copy each buffered run with one reservation.
            let count = self
                .source
                .fill_buf()?
                .iter()
                .take_while(|&&byte| (0x20..=0x7e).contains(&byte) && !matches!(byte, b'"' | b'\\'))
                .count();
            if count != 0 {
                self.charge_unit(count)?;
                self.work.reserve_string(&mut out, count, self.kind)?;
                out.push_str(std::str::from_utf8(&self.source.fill_buf()?[..count]).unwrap());
                self.consume(count)?;
                continue;
            }
            let byte = self.next()?.unwrap();
            let ch = match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self
                        .next()?
                        .ok_or_else(|| json_error("unterminated escape"))?;
                    match escaped {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' if !self.canonical => '/',
                        b'b' => '\u{08}',
                        b'f' => '\u{0c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let ch = self.parse_unicode_escape()?;
                            if self.canonical
                                && (!ch.is_control()
                                    || matches!(ch, '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}'))
                            {
                                return Err(canonical_error());
                            }
                            ch
                        }
                        b'/' => return Err(canonical_error()),
                        other => {
                            return Err(json_error(format!(
                                "invalid escape '\\{}' at {}",
                                other as char, self.pos
                            )));
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(json_error(format!(
                        "control character in string at {}",
                        self.pos
                    )));
                }
                byte if byte < 0x80 => byte as char,
                _ => {
                    let width = match byte {
                        0xc2..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf4 => 4,
                        _ => return Err(json_error("invalid utf-8 leading byte")),
                    };
                    let mut bytes = [0u8; 4];
                    bytes[0] = byte;
                    for slot in &mut bytes[1..width] {
                        *slot = self
                            .next()?
                            .ok_or_else(|| json_error("incomplete utf-8 character"))?;
                    }
                    let s = std::str::from_utf8(&bytes[..width])
                        .map_err(|err| json_error(format!("invalid utf-8: {err}")))?;
                    let ch = s
                        .chars()
                        .next()
                        .ok_or_else(|| json_error("invalid utf-8"))?;
                    ch
                }
            };
            if self.canonical && byte != b'\\' && ch.is_control() {
                return Err(canonical_error());
            }
            self.charge_unit(char_canonical_bytes(ch))?;
            self.work
                .reserve_string(&mut out, ch.len_utf8(), self.kind)?;
            out.push(ch);
        }
        Err(json_error("unterminated string"))
    }

    fn parse_unicode_escape(&mut self) -> Result<char> {
        let first = self.parse_hex_escape()?;
        let value = match first {
            0xd800..=0xdbff => {
                self.expect_byte(b'\\')?;
                self.expect_byte(b'u')?;
                let second = self.parse_hex_escape()?;
                if !(0xdc00..=0xdfff).contains(&second) {
                    return Err(json_error("invalid unicode surrogate pair"));
                }
                0x10000 + ((first - 0xd800) << 10) + second - 0xdc00
            }
            0xdc00..=0xdfff => return Err(json_error("unpaired unicode surrogate")),
            value => value,
        };
        char::from_u32(value).ok_or_else(|| json_error("invalid unicode code point"))
    }

    fn parse_hex_escape(&mut self) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .next()?
                .ok_or_else(|| json_error("unterminated unicode escape"))?;
            value = value * 16
                + match byte {
                    b'0'..=b'9' => (byte - b'0') as u32,
                    b'a'..=b'f' => (byte - b'a' + 10) as u32,
                    b'A'..=b'F' if !self.canonical => (byte - b'A' + 10) as u32,
                    b'A'..=b'F' => return Err(canonical_error()),
                    _ => return Err(json_error("invalid unicode escape")),
                };
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        let mut raw = NumberToken::default();
        if self.peek()? == Some(b'-') {
            self.number_byte(&mut raw)?;
        }
        match self.peek()? {
            Some(b'0') => self.number_byte(&mut raw)?,
            Some(b'1'..=b'9') => {
                while matches!(self.peek()?, Some(b'0'..=b'9')) {
                    self.number_byte(&mut raw)?;
                }
            }
            _ => return Err(json_error(format!("invalid number at {start}"))),
        }
        if self.peek()? == Some(b'.') {
            self.number_byte(&mut raw)?;
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(json_error(format!("invalid number at {start}")));
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.number_byte(&mut raw)?;
            }
        }
        if matches!(self.peek()?, Some(b'e' | b'E')) {
            self.number_byte(&mut raw)?;
            if matches!(self.peek()?, Some(b'+' | b'-')) {
                self.number_byte(&mut raw)?;
            }
            if !matches!(self.peek()?, Some(b'0'..=b'9')) {
                return Err(json_error(format!("invalid exponent at {start}")));
            }
            while matches!(self.peek()?, Some(b'0'..=b'9')) {
                self.number_byte(&mut raw)?;
            }
        }
        validate_integer_precision(raw.as_str())?;
        let value = raw
            .as_str()
            .parse::<f64>()
            .map_err(|_| json_error("invalid JSON number"))?;
        if !value.is_finite() {
            return Err(json_error("JSON number is not finite"));
        }
        if self.canonical && !is_canonical_number(value, raw.as_str()) {
            return Err(canonical_error());
        }
        let capacity = raw.spill.capacity();
        drop(raw);
        self.work.release(capacity);
        Ok(JsonValue::Number(value))
    }

    fn number_byte(&mut self, raw: &mut NumberToken) -> Result<()> {
        let byte = self
            .next()?
            .ok_or_else(|| json_error("incomplete number"))?;
        if raw.len < raw.inline.len() {
            raw.inline[raw.len] = byte;
        } else {
            if raw.spill.is_empty() {
                self.work
                    .reserve_string(&mut raw.spill, raw.len + 1, self.kind)?;
                raw.spill
                    .push_str(std::str::from_utf8(&raw.inline).unwrap());
            } else {
                self.work.reserve_string(&mut raw.spill, 1, self.kind)?;
            }
            raw.spill.push(char::from(byte));
        }
        raw.len += 1;
        Ok(())
    }

    fn expect_literal(&mut self, literal: &[u8]) -> Result<()> {
        for &byte in literal {
            if self.next()? != Some(byte) {
                return Err(json_error(format!("expected literal at {}", self.pos)));
            }
        }
        Ok(())
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        match self.next()? {
            Some(byte) if byte == expected => Ok(()),
            Some(byte) => Err(json_error(format!(
                "expected '{}', got '{}' at {}",
                expected as char,
                byte as char,
                self.pos.saturating_sub(1)
            ))),
            None => Err(json_error(format!("expected '{}'", expected as char))),
        }
    }

    fn consume_if(&mut self, expected: u8) -> Result<bool> {
        if self.peek()? == Some(expected) {
            self.consume(1)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn skip_ws(&mut self) -> Result<()> {
        while matches!(self.peek()?, Some(b' ' | b'\n' | b'\r' | b'\t')) {
            if self.canonical {
                return Err(canonical_error());
            }
            let count = self
                .source
                .fill_buf()?
                .iter()
                .take_while(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
                .count();
            self.consume(count)?;
        }
        Ok(())
    }

    fn peek(&mut self) -> Result<Option<u8>> {
        loop {
            match self.source.fill_buf() {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => return Ok(result?.first().copied()),
            }
        }
    }

    fn consume(&mut self, count: usize) -> Result<()> {
        self.pos = self
            .pos
            .checked_add(count)
            .ok_or_else(|| json_error("JSON input offset exceeds supported range"))?;
        self.source.consume(count);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<u8>> {
        let byte = self.peek()?;
        if byte.is_some() {
            self.consume(1)?;
        }
        Ok(byte)
    }
}

struct NumberToken {
    inline: [u8; 128],
    len: usize,
    spill: String,
}

impl Default for NumberToken {
    fn default() -> Self {
        Self {
            inline: [0; 128],
            len: 0,
            spill: String::new(),
        }
    }
}

impl NumberToken {
    fn as_str(&self) -> &str {
        if self.spill.is_empty() {
            std::str::from_utf8(&self.inline[..self.len]).unwrap()
        } else {
            &self.spill
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_json_parse_and_copy_failures_preserve_classification_input_and_credit() {
        for kind in [
            JsonInputKind::JsonlRecord,
            JsonInputKind::Filter,
            JsonInputKind::StoredJson,
        ] {
            let bytes = 131_071;
            let value = JsonValue::String("x".repeat(bytes));
            let wire = format!("\"{}\"", value.as_str().unwrap());
            let context = JsonOperationContext::standalone();
            let capacity = context.available_workspace_bytes();
            let input_work = value.reserve_retained_in(kind, &context).unwrap();
            let baseline = context.available_workspace_bytes();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                parse_with_schema_in(&wire, JsonSchema::Value(kind), &context)
            })
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert!(!is_memory_budget_error(&error));
            assert_eq!(context.available_workspace_bytes(), baseline);
            let parsed = parse_with_schema_in(&wire, JsonSchema::Value(kind), &context).unwrap();
            assert_eq!(parsed.value, value);
            drop(parsed);
            assert_eq!(context.available_workspace_bytes(), baseline);
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                value.try_clone_in(kind, &context)
            })
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(context.available_workspace_bytes(), baseline);
            let copy = value.try_clone_in(kind, &context).unwrap();
            assert_eq!(copy.value, value);
            drop(copy);
            let mut copy_work = context.reservation();
            copy_work.grow(17, kind).unwrap();
            let error = crate::test_alloc::fail_byte_allocation_once(bytes, || {
                copy_work.copy_validated_value(&value, kind)
            })
            .unwrap_err();
            assert_eq!(error.code(), ErrorCode::IoError);
            assert!(error.detail().is_empty());
            assert_eq!(copy_work.reserved_bytes(), 17);
            assert_eq!(context.available_workspace_bytes(), baseline - 17);
            let copy = copy_work.copy_validated_value(&value, kind).unwrap();
            assert_eq!(copy, value);
            assert_eq!(copy_work.reserved_bytes(), bytes + 17);
            drop(copy);
            drop(copy_work);
            assert_eq!(context.available_workspace_bytes(), baseline);
            assert_eq!(value.as_str().unwrap().len(), bytes);
            drop(value);
            drop(input_work);
            assert_eq!(context.available_workspace_bytes(), capacity);
        }
    }

    #[test]
    fn retained_layout_spare_does_not_block_necessary_stack_growth() {
        let mut value = JsonValue::Null;
        for _ in 0..5 {
            value = JsonValue::Array(vec![value]);
        }
        let bytes = 5 * std::mem::size_of::<JsonValue>();
        for parent in [false, true] {
            for limit in (4096..=5120).step_by(32) {
                let ledger = std::sync::Arc::new(WorkLedger::with_limits(
                    if parent { 8192 } else { limit },
                    0,
                    if parent { limit } else { 8192 },
                ));
                let context = JsonOperationContext::new(ledger);
                let (retained, canonical) =
                    value.guard_in(JsonInputKind::StoredJson, &context).unwrap();
                assert_eq!(retained.reserved_bytes(), bytes);
                assert_eq!(canonical, 14);
                assert_eq!(context.available_workspace_bytes(), limit - bytes);
                drop(retained);
                assert_eq!(context.available_workspace_bytes(), limit);
            }
        }
    }

    #[test]
    fn retained_handoff_reuses_credit_and_releases_only_discarded_allocations() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let mut value = object_from_pairs([
            ("keep", JsonValue::String("x".repeat(6 * 1024 * 1024))),
            ("discard", JsonValue::String("y".repeat(1024 * 1024))),
        ]);
        let mut work = value.reserve_retained_in(kind, &context).unwrap();
        let mut other = context.reservation();
        other
            .grow(
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - work.reserved_bytes() - 4096,
                kind,
            )
            .unwrap();
        // A second complete charge cannot fit, but transferring the same nodes can.
        assert!(value.reserve_retained_in(kind, &context).is_err());
        value.reconcile_retained_in(kind, &mut work).unwrap();
        let previous = work.reserved_bytes();
        drop(value.as_object_mut().unwrap().remove("discard"));
        value.reconcile_retained_in(kind, &mut work).unwrap();
        assert_eq!(
            previous - work.reserved_bytes(),
            1024 * 1024 + "discard".len()
        );
        let mut released = context.reservation();
        released.grow(1024 * 1024, kind).unwrap();
        drop(released);
        drop(other);
        let measured = value.reserve_retained_in(kind, &context).unwrap();
        assert_eq!(measured.reserved_bytes(), work.reserved_bytes());
    }

    #[test]
    fn retained_handoff_charges_scratch_and_does_not_release_credit_on_error() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let value = object_from_pairs([("payload", JsonValue::String("x".repeat(4096)))]);
        let mut work = value.reserve_retained_in(kind, &context).unwrap();
        let previous = work.reserved_bytes();
        let mut other = context.reservation();
        other
            .grow(JsonProcessingPolicyV1::WORK_BUDGET_BYTES - previous, kind)
            .unwrap();
        assert!(value.reconcile_retained_in(kind, &mut work).is_err());
        assert_eq!(work.reserved_bytes(), previous);
        drop(other);
        value.reconcile_retained_in(kind, &mut work).unwrap();
        assert_eq!(work.reserved_bytes(), previous);
        drop(value);
        drop(work);
        context
            .reservation()
            .grow(JsonProcessingPolicyV1::WORK_BUDGET_BYTES, kind)
            .unwrap();
    }

    struct Fragmented<'a> {
        bytes: &'a [u8],
        fragment: usize,
        interrupt: bool,
        consumed: usize,
    }

    impl Read for Fragmented<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(buffer.len() <= 8192);
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let count = buffer.len().min(self.fragment).min(self.bytes.len());
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            self.consumed += count;
            Ok(count)
        }
    }

    #[test]
    fn typed_object_guard_counts_enclosing_depth_and_canonical_field_bytes() {
        let context = JsonOperationContext::standalone();
        let kind = JsonInputKind::CollectionConfig;
        let text = "quote\"\n\u{85}é".to_owned();
        let child = parse("[null,1,{}]").unwrap();
        let mut guard = JsonObjectGuard::new(kind, &context);
        guard.string_field("name", &text).unwrap();
        guard.value_field("child", &child).unwrap();
        let complete = object_from_pairs([("name", JsonValue::String(text)), ("child", child)]);
        let measured = complete.reserve_retained_in(kind, &context).unwrap();
        assert_eq!(guard.work.reserved_bytes(), measured.reserved_bytes());
        drop(measured);
        assert_eq!(guard.bytes, complete.to_compact_string().len());
        drop(guard);
        for (depth, accepted) in [(127, true), (128, false)] {
            let value = parse(&mixed_json(depth, "0")).unwrap();
            let mut guard = JsonObjectGuard::new(kind, &context);
            let result = guard.value_field("nested", &value);
            assert_eq!(result.is_ok(), accepted);
            if let Err(error) = result {
                assert!(is_depth_error(&error));
            }
        }
    }

    #[test]
    fn typed_object_guard_retains_every_fields_workspace() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let kind = JsonInputKind::CollectionConfig;
            let name = "n".repeat(seed * 17);
            let first = JsonValue::Array(vec![JsonValue::Null; seed]);
            let second = object_from_pairs([("text", JsonValue::String("x".repeat(seed * 31)))]);
            let mut guard = JsonObjectGuard::new(kind, &context);
            guard.string_field("name", &name).unwrap();
            guard.value_field("first", &first).unwrap();
            guard.value_field("second", &second).unwrap();
            let complete = object_from_pairs([
                ("name", JsonValue::String(name)),
                ("first", first),
                ("second", second),
            ]);
            let measured = complete.reserve_retained_in(kind, &context).unwrap();
            assert_eq!(guard.work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn bounded_canonical_input_reserves_its_frame_size_and_preserves_the_next_field() {
        struct Checked<'a> {
            source: Cursor<&'a [u8]>,
            context: &'a JsonOperationContext,
            buffer_bytes: usize,
            fragment: usize,
            first: bool,
        }
        impl Read for Checked<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                assert!(output.len() <= self.buffer_bytes);
                if self.first {
                    self.first = false;
                    let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
                    let mut other = self.context.workspace_reservation();
                    other
                        .grow_to(
                            limit - self.buffer_bytes,
                            limit - self.buffer_bytes,
                            "operation_scratch",
                        )
                        .unwrap();
                    assert!(other
                        .grow_to(
                            limit - self.buffer_bytes + 1,
                            limit - self.buffer_bytes + 1,
                            "operation_scratch"
                        )
                        .is_err());
                }
                let count = output.len().min(self.fragment);
                self.source.read(&mut output[..count])
            }
        }

        for seed in 0..100usize {
            let value = if seed % 4 == 0 {
                JsonValue::object()
            } else {
                object_from_pairs([("value", JsonValue::String("\u{4e2d}\n".repeat(seed * 31)))])
            };
            let canonical = value.to_compact_string();
            let mut wire = canonical.as_bytes().to_vec();
            wire.extend_from_slice(b"next field");
            let context = JsonOperationContext::standalone();
            let mut source = Checked {
                source: Cursor::new(wire.as_slice()),
                context: &context,
                buffer_bytes: canonical.len().min(8192),
                fragment: 1 + seed * 17,
                first: true,
            };
            let mut limited = (&mut source).take(canonical.len() as u64);
            let parsed =
                parse_canonical_reader_in(&mut limited, JsonInputKind::StoredJson, &context)
                    .unwrap();
            assert_eq!(limited.limit(), 0);
            assert_eq!(parsed.value, value);
            assert_eq!(source.source.position(), canonical.len() as u64);
            assert!(!source.first);
            let mut other = context.workspace_reservation();
            let remaining =
                JsonProcessingPolicyV1::WORK_BUDGET_BYTES - parsed._reservation.reserved_bytes();
            other
                .grow_to(remaining, remaining, "operation_scratch")
                .unwrap();
            drop(parsed);
            other
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }

    #[test]
    fn bounded_canonical_input_keeps_json_limits_and_releases_failed_buffers() {
        let kind = JsonInputKind::StoredJson;
        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100usize {
            let context = JsonOperationContext::standalone();
            let mut held = context.workspace_reservation();
            held.grow_to(limit - 512, limit - 512, "operation_scratch")
                .unwrap();
            let mut small = Cursor::new(b"{}next").take(2);
            assert!(parse_canonical_reader_in(&mut small, kind, &context)
                .unwrap()
                .value
                .as_object()
                .unwrap()
                .is_empty());
            assert_eq!(small.into_inner().position(), 2);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
            for raw in [b"{}".as_slice(), b"{\"a\":1}".as_slice()] {
                let mut source = Cursor::new(raw).take(raw.len() as u64);
                let error = parse_canonical_reader_in(&mut source, kind, &context)
                    .err()
                    .unwrap();
                assert_eq!(error.code(), ErrorCode::RecoveryError);
                assert_eq!(source.get_ref().position(), 0);
                assert!(is_limit_error(&error));
                assert!(error
                    .detail()
                    .iter()
                    .any(|(key, value)| key == "limit" && value == "0"));
            }
            held.retain(0);
            for raw in [
                "".to_owned(),
                "{\"a\":1".to_owned(),
                "{\"b\":1,\"a\":2}".to_owned(),
                "{\"a\":0,\"a\":1}".to_owned(),
                format!("{}0{}", "[".repeat(129 + seed), "]".repeat(129 + seed)),
            ] {
                let mut source = Cursor::new(raw.as_bytes()).take(raw.len() as u64);
                assert!(parse_canonical_reader_in(&mut source, kind, &context).is_err());
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                held.retain(0);
            }
        }
    }

    #[test]
    fn streaming_parser_preserves_syntax_precision_and_split_unicode() {
        let long_number = format!("1.{}1", "0".repeat(150));
        for raw in [
            r#" {"a":[null,true,false,-1.25e-20],"b":"é存储\ud83d\ude80\n\u0085"} "#,
            r#"{"a":0,"\u0061":1}"#,
            "[1,]",
            "1e+",
            "9007199254740993",
            "9007199254740993.000000000000000000000000000000000000",
            "[\"\\ud800\"]",
            &long_number,
        ] {
            let expected = parse(raw);
            for fragment in [1, 2, 3, 7, 127, 8192] {
                let mut source = Fragmented {
                    bytes: raw.as_bytes(),
                    fragment,
                    interrupt: false,
                    consumed: 0,
                };
                let actual =
                    parse_reader(&mut source, JsonSchema::Value(JsonInputKind::JsonlRecord));
                match (&expected, actual) {
                    (Ok(expected), Ok(actual)) => assert_eq!(expected, &actual),
                    (Err(expected), Err(actual)) => assert_eq!(expected.code(), actual.code()),
                    (_, actual) => panic!("fragment={fragment}, input={raw:?}, result={actual:?}"),
                }
            }
        }
        for bytes in [
            &b"\"\xc0\x80\""[..],
            &b"\"\xed\xa0\x80\""[..],
            &b"\"\xf0\x90"[..],
        ] {
            assert!(parse_reader(bytes, JsonSchema::Value(JsonInputKind::Manifest)).is_err());
        }
    }

    #[test]
    fn streaming_parser_bounds_prefetch_and_does_not_count_wire_whitespace() {
        let whitespace = std::io::repeat(b' ').take(40 * 1_048_576);
        let source = whitespace.chain(Cursor::new(b"{\"a\":1}\n\t"));
        assert_eq!(
            parse_reader(source, JsonSchema::Value(JsonInputKind::Manifest)).unwrap(),
            parse("{\"a\":1}").unwrap()
        );
        let raw = "[".repeat(100_000);
        let mut source = Fragmented {
            bytes: raw.as_bytes(),
            fragment: 8192,
            interrupt: false,
            consumed: 0,
        };
        let error =
            parse_reader(&mut source, JsonSchema::Value(JsonInputKind::Manifest)).unwrap_err();
        assert!(is_depth_error(&error));
        assert_eq!(source.consumed, 8192);
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::Other.into())
            }
        }
        assert_eq!(
            parse_reader(Broken, JsonSchema::Value(JsonInputKind::Manifest))
                .unwrap_err()
                .code(),
            ErrorCode::IoError
        );
    }

    #[test]
    fn streamed_canonical_validation_and_digest_match_the_writer() {
        let mut seed = 0x6a09_e667_f3bc_c909u64;
        for round in 0..100 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let number = f64::from_bits(seed);
            let number = if number.is_finite() { number } else { 0.0 };
            let value = object_from_pairs([
                (
                    "a",
                    JsonValue::Array(vec![
                        JsonValue::Number(number),
                        JsonValue::Number(-0.0),
                        JsonValue::Bool(round % 2 == 0),
                    ]),
                ),
                (
                    "b\n\u{7f}存储",
                    JsonValue::String(format!(
                        "{round}:\"\\/\u{0}\u{08}\u{0c}\r\n\t\u{7f}\u{85}é存储"
                    )),
                ),
            ]);
            let canonical = value.to_compact_string();
            assert_eq!(
                value.canonical_digest(JsonInputKind::StoredJson).unwrap(),
                (
                    canonical.len(),
                    crate::hash::sha256_hex(canonical.as_bytes())
                )
            );
            for fragment in [1, 7, 8192] {
                let source = Fragmented {
                    bytes: canonical.as_bytes(),
                    fragment,
                    interrupt: false,
                    consumed: 0,
                };
                assert_eq!(
                    parse_canonical_reader(source, JsonInputKind::StoredJson).unwrap(),
                    value
                );
            }
        }
        for raw in [
            " {}",
            "{}\n",
            "{\"a\" :0}",
            "{\"b\":0,\"a\":1}",
            "-0",
            "0.0",
            "1e0",
            "1.00",
            "1E+1",
            r#""\/""#,
            r#""\u0061""#,
            r#""\u000a""#,
            r#""\u007F""#,
            r#""\ud83d\ude80""#,
            "\"\u{7f}\"",
            "\"\u{85}\"",
        ] {
            assert!(parse(raw).is_ok(), "{raw:?}");
            assert_eq!(
                parse_canonical_reader(raw.as_bytes(), JsonInputKind::StoredJson)
                    .unwrap_err()
                    .code(),
                ErrorCode::FileFormatError,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn canonical_unit_bytes_count_escaping_keys_and_not_wire_whitespace() {
        let limit = JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES;
        let exact = JsonValue::String("x".repeat(limit - 2));
        assert!(exact.validate_kind(JsonInputKind::DocumentMetadata).is_ok());
        let raw = format!("  {}\n\t", exact.to_compact_string());
        assert_eq!(
            parse_with_kind(&raw, JsonInputKind::DocumentMetadata).unwrap(),
            exact
        );
        let too_large = JsonValue::String("x".repeat(limit - 1));
        for error in [
            too_large
                .validate_kind(JsonInputKind::DocumentMetadata)
                .unwrap_err(),
            parse_with_kind(
                &too_large.to_compact_string(),
                JsonInputKind::DocumentMetadata,
            )
            .unwrap_err(),
        ] {
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "unit_bytes_exceeded"));
            assert_eq!(
                error
                    .to_json()
                    .get("detail")
                    .unwrap()
                    .get("actual")
                    .and_then(JsonValue::as_f64),
                Some((limit + 1) as f64)
            );
        }
        let escaped =
            object_from_pairs([("\"\\\t\u{7f}\u{85}", JsonValue::String("é\n\"".into()))]);
        let root = shallow_canonical_bytes(&escaped);
        let leaf = escaped.as_object().unwrap().values().next().unwrap();
        assert_eq!(
            root + shallow_canonical_bytes(leaf),
            escaped.to_compact_string().len()
        );
        assert!(parse_with_kind(
            &format!("{}0", " ".repeat(limit + 1)),
            JsonInputKind::Manifest
        )
        .is_ok());
    }

    #[test]
    fn record_schema_does_not_charge_outer_fields_to_metadata() {
        for (schema, kind) in [
            (JsonSchema::DocumentRow, JsonInputKind::DocumentMetadata),
            (JsonSchema::ChunkRow, JsonInputKind::ChunkMetadata),
            (JsonSchema::StoredRecord, JsonInputKind::StoredJson),
        ] {
            let raw = format!("{{\"metadata\":{}}}", mixed_json(MAX_JSON_DEPTH, "0"));
            assert!(parse_with_schema(&raw, schema).is_ok());
            let raw = format!("{{\"metadata\":{}}}", mixed_json(MAX_JSON_DEPTH + 1, "0"));
            let error = parse_with_schema(&raw, schema).unwrap_err();
            assert_eq!(error.code(), kind.error_code());
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == kind.as_str()));
        }
        let text = "x".repeat(JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES + 1);
        assert!(parse_with_schema(
            &format!("{{\"text\":\"{text}\",\"metadata\":{{}}}}"),
            JsonSchema::ChunkRow
        )
        .is_ok());
        let raw = format!("{{\"metadata\":\"{text}\"}}");
        assert!(is_limit_error(
            &parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err()
        ));
        // An unknown nested wrapper is not a recognized schema boundary.
        let raw = format!("{{\"unknown\":{}}}", mixed_json(MAX_JSON_DEPTH + 1, "0"));
        assert!(is_depth_error(
            &parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err()
        ));
    }

    #[test]
    fn query_vector_canonical_bytes_use_the_same_f32_values_for_text_and_typed_inputs() {
        for raw in ["[0.1,0.2,-0.0,1e-50]", "[1.00000001,0.125,-8388608]"] {
            let mut parser = Parser {
                source: Cursor::new(raw.as_bytes()),
                canonical: false,
                pos: 0,
                unit: None,
                kind: JsonInputKind::QueryEmbedding,
                work: JsonOperationContext::standalone().reservation(),
            };
            let value = parser
                .parse_value(0, JsonSchema::Value(JsonInputKind::QueryEmbedding))
                .unwrap();
            let typed: Vec<_> = value
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            validate_query_vector(&typed).unwrap();
            let normalized = JsonValue::Array(
                typed
                    .into_iter()
                    .map(|value| JsonValue::Number(f64::from(value)))
                    .collect(),
            );
            assert_eq!(parser.unit.unwrap().1, normalized.to_compact_string().len());
        }
        let error = parse_with_kind(
            "[340282366920938463463374607431768211456]",
            JsonInputKind::QueryEmbedding,
        )
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidVectorLength);
    }

    #[test]
    fn parser_uses_explicit_stack_on_small_host_threads_and_rejects_duplicates() {
        std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(|| {
                assert!(parse(&mixed_json(MAX_JSON_DEPTH, "0")).is_ok());
                assert!(is_depth_error(&parse(&"[".repeat(30_000)).unwrap_err()));
                for raw in [
                    r#"{"a":null,"\u0061":1}"#,
                    "[1,]",
                    "{\"a\":0,}",
                    "[[] 0]",
                    "{}[]",
                ] {
                    assert!(parse(raw).is_err(), "{raw}");
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn work_budget_limits_wide_values_without_confusing_them_with_unit_bytes() {
        let values = JsonValue::Array(vec![JsonValue::Null; 1_100_000]);
        let error = values
            .validate_kind(JsonInputKind::DocumentMetadata)
            .unwrap_err();
        let detail = error.to_json();
        assert_eq!(
            detail
                .get("detail")
                .unwrap()
                .get("json_limit_reason")
                .unwrap()
                .as_str(),
            Some("work_budget_exceeded")
        );
        let raw = format!("[{}null]", "null,".repeat(1_100_000));
        let error = parse_with_kind(&raw, JsonInputKind::DocumentMetadata).unwrap_err();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        assert!(raw.len() < JsonProcessingPolicyV1::MAX_UNIT_CANONICAL_BYTES);
    }

    #[test]
    fn jsonl_outer_text_work_limit_uses_record_kind_but_metadata_keeps_its_kind() {
        for (field, expected) in [
            ("text", JsonInputKind::JsonlRecord),
            ("metadata", JsonInputKind::ChunkMetadata),
        ] {
            let raw = if field == "text" {
                format!(
                    "{{\"metadata\":{{}},\"text\":\"{}\"}}",
                    "x".repeat(JsonProcessingPolicyV1::WORK_BUDGET_BYTES)
                )
            } else {
                format!(
                    "{{\"metadata\":[{}null],\"text\":\"x\"}}",
                    "null,".repeat(1_100_000)
                )
            };
            let error = parse_with_schema(&raw, JsonSchema::ChunkRow).unwrap_err();
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_input_kind" && value == expected.as_str()));
            assert!(error
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        }
    }

    #[test]
    fn parsed_values_hold_one_operation_budget_until_their_owned_memory_is_released() {
        let context = JsonOperationContext::standalone();
        let raw = format!("[{}null]", "null,".repeat(600_000));
        let first =
            parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context).unwrap();
        let error = parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context)
            .err()
            .unwrap();
        assert!(error
            .detail()
            .iter()
            .any(|(key, value)| key == "json_limit_reason" && value == "work_budget_exceeded"));
        drop(first);
        assert!(
            parse_with_schema_in(&raw, JsonSchema::Value(JsonInputKind::Filter), &context).is_ok()
        );
    }

    #[test]
    fn number_byte_measurement_agrees_with_serialization_at_binary64_extremes() {
        for value in [
            0.0,
            -0.0,
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            0.1,
            -0.1,
        ] {
            let json = JsonValue::Number(value);
            assert_eq!(
                number_canonical_bytes(value),
                json.to_compact_string().len()
            );
        }
    }

    fn mixed_json(depth: usize, leaf: &str) -> String {
        let mut prefix = String::new();
        let mut suffix = String::new();
        for n in 0..depth {
            if n % 2 == 0 {
                prefix.push_str("{\"v\":");
                suffix.push('}');
            } else {
                prefix.push('[');
                suffix.push(']');
            }
        }
        format!("{prefix}{leaf}{}", suffix.chars().rev().collect::<String>())
    }

    #[test]
    fn container_depth_counts_empty_and_mixed_containers_but_not_scalars() {
        for leaf in ["0", "null", "\"unicode \\ud83d\\ude80\""] {
            let raw = mixed_json(MAX_JSON_DEPTH, leaf);
            let value = parse(&raw).unwrap();
            assert!(value.validate_kind(JsonInputKind::JsonlRecord).is_ok());
            assert!(value.try_to_compact_string().is_ok());
            let err = parse(&mixed_json(MAX_JSON_DEPTH + 1, leaf)).unwrap_err();
            assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "json_limit_reason" && value == "depth_exceeded"));
            assert!(err
                .detail()
                .iter()
                .any(|(key, value)| key == "limit" && value == "128"));
        }
        assert!(parse(&mixed_json(MAX_JSON_DEPTH - 1, "{}")).is_ok());
        assert!(parse(&mixed_json(MAX_JSON_DEPTH, "[]")).is_err());
        assert!(parse(&"[".repeat(30_000)).is_err());
        assert!(parse(&format!("{}!{}", "[".repeat(30_000), "]".repeat(30_000))).is_err());
        assert!(parse(&mixed_json(MAX_JSON_DEPTH, "!")).is_err());
    }

    #[test]
    fn wide_shallow_json_and_pretty_output_remain_compatible() {
        let obj = (0..20_000)
            .map(|n| {
                (
                    format!("key-{n:05}"),
                    JsonValue::Array(vec![JsonValue::Number(n as f64)]),
                )
            })
            .collect();
        let value = JsonValue::Object(obj);
        let raw = value.try_to_compact_string().unwrap();
        assert_eq!(parse(&raw).unwrap(), value);
        let value = parse(r#"{"a":[1,{"b":true}],"empty":{}}"#).unwrap();
        assert_eq!(
            value.to_pretty_string(),
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": true\n    }\n  ],\n  \"empty\": {}\n}"
        );
        assert_eq!(value.clone(), value);
        assert_ne!(JsonValue::Number(f64::NAN), JsonValue::Number(f64::NAN));
    }

    #[test]
    fn fixed_seed_json_syntax_smoke_never_panics() {
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let alphabet = b"{}[],:\"\\ntf0123456789-+e abc";
        for _ in 0..2000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = seed as usize % 512;
            let mut raw = String::new();
            for _ in 0..len {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                raw.push(alphabet[seed as usize % alphabet.len()] as char);
            }
            if let Ok(value) = parse(&raw) {
                let canonical = value.try_to_compact_string().unwrap();
                assert_eq!(parse(&canonical).unwrap(), value);
            }
        }
    }

    #[test]
    fn json_round_trips_objects_arrays_and_unicode() {
        let value = parse(r#"{"lang":"rust","tags":["db",2],"ok":true,"u":"存储"}"#).unwrap();
        assert_eq!(value.get("lang").and_then(JsonValue::as_str), Some("rust"));
        let encoded = value.to_compact_string();
        assert_eq!(parse(&encoded).unwrap(), value);
    }

    #[test]
    fn rejects_non_finite_json_numbers() {
        assert!(parse(r#"{"too_big":1e999}"#).is_err());
        assert!(JsonValue::Number(f64::NAN).try_to_compact_string().is_err());
    }

    #[test]
    fn rejects_duplicate_object_keys() {
        let err = parse(r#"{"metadata":{"lang":"rust","lang":"go"}}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert_eq!(err.message(), "duplicate object key");
    }

    #[test]
    fn rejects_integer_precision_loss_without_rejecting_exact_large_values() {
        let err = parse(r#"{"metadata":{"id":9007199254740993}}"#).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidJsonlSchema);
        assert!(err.message().contains("encode it as a string"));
        assert!(parse(r#"{"metadata":{"id":9007199254740991}}"#).is_ok());
        for raw in [
            "9007199254740992",
            "9007199254740994",
            "-9007199254740992",
            "9007199254740992.00",
            "90071992547409920e-1",
            "9.007199254740992E+15",
            "9223372036854775808",
            "18446744073709551616",
            "1267650600228229401496703205376",
        ] {
            let value = parse(raw).unwrap();
            assert_eq!(parse(&value.to_compact_string()).unwrap(), value, "{raw}");
        }
        for raw in [
            "-9007199254740993",
            "9007199254740993.0000",
            "90071992547409930e-1",
            "9.007199254740993e15",
            "18446744073709551615",
            "1267650600228229401496703205377",
            "1e309",
            "1e999999999999999999999",
        ] {
            assert!(parse(raw).is_err(), "{raw}");
        }
        assert!(parse("9007199254740993.5").is_ok());
        assert!(parse("1e-999999999999999999999").is_ok());
        assert_eq!(parse("-0e999999").unwrap(), JsonValue::Number(0.0));
        assert_eq!(
            parse(&format!("{:.0}", f64::MAX)).unwrap(),
            JsonValue::Number(f64::MAX)
        );
    }

    #[test]
    fn integer_precision_matches_independent_u128_bit_boundaries() {
        for exponent in 53..128 {
            let base = 1u128 << exponent;
            for offset in 0..8 {
                let value = base + offset;
                let exact = 128 - value.leading_zeros() - value.trailing_zeros() <= 53;
                for sign in ["", "-"] {
                    for suffix in ["", ".0", "00e-2"] {
                        let raw = format!("{sign}{value}{suffix}");
                        assert_eq!(parse(&raw).is_ok(), exact, "{raw}");
                    }
                }
            }
        }
    }

    #[test]
    fn canonical_bytes_have_independent_utf8_key_and_escaping_golden() {
        let input =
            "{\"\u{10000}\":1,\"\u{e000}\":2,\"a\":[-0,3],\"s\":\"a\\\"\\\\\\n\\t\\u0000\"}";
        let expected =
            "{\"a\":[0,3],\"s\":\"a\\\"\\\\\\n\\t\\u0000\",\"\u{e000}\":2,\"\u{10000}\":1}";
        assert_eq!(parse(input).unwrap().to_compact_string(), expected);
    }

    #[test]
    fn unicode_surrogate_pairs_decode_to_scalar_values_and_duplicate_keys() {
        for (raw, expected) in [
            (r#""\ud800\udc00""#, "\u{10000}"),
            (r#""\uDBFF\uDFFF""#, "\u{10ffff}"),
            (r#""\ud83d\ude80""#, "\u{1f680}"),
        ] {
            let value = parse(raw).unwrap();
            assert_eq!(value, JsonValue::String(expected.into()));
            assert_eq!(parse(&value.to_compact_string()).unwrap(), value);
        }
        for raw in [
            r#""\ud800""#,
            r#""\udc00""#,
            r#""\ud800\u0041""#,
            r#""\ud800\ud800""#,
            r#""\ud800x""#,
            r#""\ud800\u""#,
        ] {
            assert!(parse(raw).is_err(), "{raw}");
        }
        let raw = "{\"\u{10000}\":1,\"\\ud800\\udc00\":2}";
        assert!(parse(raw)
            .unwrap_err()
            .message()
            .contains("duplicate object key"));
    }

    #[test]
    fn usize_conversion_rejects_exclusive_bound_instead_of_saturating() {
        let exclusive_max = (usize::MAX as u128 + 1) as f64;
        let largest_integer = if usize::BITS <= 53 {
            exclusive_max - 1.0
        } else {
            f64::from_bits(exclusive_max.to_bits() - 1)
        };
        let value = object_from_pairs([("count", JsonValue::Number(largest_integer))]);
        assert_eq!(
            required_usize(value.as_object().unwrap(), "count").unwrap(),
            largest_integer as usize
        );
        for invalid in [exclusive_max, exclusive_max * 2.0, -1.0, 0.5, f64::NAN] {
            let value = object_from_pairs([("count", JsonValue::Number(invalid))]);
            assert_eq!(
                required_usize(value.as_object().unwrap(), "count")
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidJsonlSchema
            );
        }
    }

    #[test]
    fn long_unicode_strings_and_keys_preserve_every_scalar() {
        let text = "\u{10000}\u{e000}\u{20ac}e".repeat(16_384);
        let input = format!("{{\"{text}\":\"{text}\"}}");
        let value = parse(&input).unwrap();
        assert_eq!(
            value.get(&text).and_then(JsonValue::as_str),
            Some(text.as_str())
        );
        assert_eq!(value.to_compact_string(), input);
    }

    #[test]
    fn canonicalizes_negative_zero_numbers() {
        assert_eq!(JsonValue::Number(-0.0).to_compact_string(), "0");
        assert_eq!(parse("-0").unwrap().to_compact_string(), "0");
        assert_eq!(
            parse(r#"{"value":-0}"#).unwrap().to_compact_string(),
            r#"{"value":0}"#
        );
    }

    #[test]
    fn retained_footprint_uses_scratch_without_readmitting_the_resident_tree() {
        for seed in 0..100 {
            let context = JsonOperationContext::standalone();
            let mut text = String::with_capacity(65536 + seed);
            text.push_str("retained");
            let value =
                object_from_pairs([("payload", JsonValue::Array(vec![JsonValue::String(text)]))]);
            let retained = value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            let expected = retained.reserved_bytes();
            let mut occupied = context.workspace_reservation();
            let bytes = JsonProcessingPolicyV1::WORK_BUDGET_BYTES - expected - 4096;
            occupied.grow_to(bytes, bytes, "operation_scratch").unwrap();
            assert_eq!(
                value
                    .retained_bytes_in(JsonInputKind::StoredJson, &context)
                    .unwrap(),
                expected
            );
            assert!(value
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .is_err());
            drop(occupied);
            drop(value);
            drop(retained);
            context
                .workspace_reservation()
                .grow_to(
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    JsonProcessingPolicyV1::WORK_BUDGET_BYTES,
                    "operation_scratch",
                )
                .unwrap();
        }
    }
}
