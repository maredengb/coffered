use std::cmp::Ordering;

use crate::catalog::Catalog;
use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::json::object_from_pairs;
use crate::json::{JsonInputKind, JsonObject, JsonOperationContext, JsonReservation, JsonValue};
mod trace;
use crate::model::Chunk;

#[derive(Clone, Debug)]
pub enum FilterExpr {
    Eq(String, JsonValue),
    Ne(String, JsonValue),
    Gt(String, JsonValue),
    Gte(String, JsonValue),
    Lt(String, JsonValue),
    Lte(String, JsonValue),
    In(String, Vec<JsonValue>),
    Contains(String, JsonValue),
    And(Vec<FilterExpr>),
}

#[derive(Clone, Debug)]
pub struct FilterExplain {
    pub matched: bool,
    pub expression: JsonValue,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MetadataBoostConfig {
    rules: Vec<MetadataBoostRule>,
    max_boost: f64,
    hash: String,
}

#[derive(Clone, Debug)]
struct MetadataBoostRule {
    when: FilterExpr,
    boost: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetadataBoostScore {
    pub boost: f64,
    pub matched_rule_count: usize,
}

impl FilterExpr {
    #[cfg(test)]
    pub fn parse(value: JsonValue) -> Result<Option<Self>> {
        let context = JsonOperationContext::standalone();
        let (_input, _) = value
            .guard_in(JsonInputKind::Filter, &context)
            .map_err(|err| err.with_detail("field", "filter"))?;
        Self::parse_validated_in(&value, &context, JsonInputKind::Filter).map(|(value, _)| value)
    }

    pub(crate) fn parse_validated_in(
        value: &JsonValue,
        context: &JsonOperationContext,
        kind: JsonInputKind,
    ) -> Result<(Option<Self>, JsonReservation)> {
        let mut work = context.reservation();
        let result = Self::parse_with_work(value, &mut work, kind)?;
        Ok((result, work))
    }

    fn parse_with_work(
        value: &JsonValue,
        work: &mut JsonReservation,
        kind: JsonInputKind,
    ) -> Result<Option<Self>> {
        struct Frame<'a> {
            fields: std::collections::btree_map::Iter<'a, String, JsonValue>,
            and_items: Option<std::slice::Iter<'a, JsonValue>>,
            nodes: Vec<FilterExpr>,
        }
        let Some(obj) = filter_object(value)? else {
            return Ok(None);
        };
        let mut frames = Vec::new();
        work.reserve_vec(&mut frames, 1, kind)?;
        frames.push(Frame {
            fields: obj.iter(),
            and_items: None,
            nodes: Vec::new(),
        });
        loop {
            let frame = frames.last_mut().unwrap();
            if let Some(item) = frame.and_items.as_mut().and_then(Iterator::next) {
                if let Some(obj) = filter_object(item)? {
                    work.reserve_vec(&mut frames, 1, kind)?;
                    frames.push(Frame {
                        fields: obj.iter(),
                        and_items: None,
                        nodes: Vec::new(),
                    });
                }
                continue;
            }
            frame.and_items = None;
            if let Some((field, condition)) = frame.fields.next() {
                if field != "$and" {
                    validate_filter_field(field)?;
                    let node = parse_field_condition(field, condition, work, kind)?;
                    work.reserve_vec(&mut frame.nodes, 1, kind)?;
                    frame.nodes.push(node);
                    continue;
                }
                let items = condition.as_array().ok_or_else(|| {
                    Error::new(ErrorCode::InvalidMetadataFilter, "$and must be an array")
                        .with_detail("field", "filter.$and")
                        .with_detail("expected", "array")
                })?;
                frame.and_items = Some(items.iter());
                continue;
            }
            let mut nodes = frames.pop().unwrap().nodes;
            let node = if nodes.len() == 1 {
                let node = nodes.pop().unwrap();
                let bytes = nodes.capacity() * std::mem::size_of::<FilterExpr>();
                drop(nodes);
                work.release(bytes);
                node
            } else {
                FilterExpr::And(nodes)
            };
            if let Some(parent) = frames.last_mut() {
                work.reserve_vec(&mut parent.nodes, 1, kind)?;
                parent.nodes.push(node);
            } else {
                let bytes = frames.capacity() * std::mem::size_of::<Frame<'_>>();
                drop(frames);
                work.release(bytes);
                return Ok(Some(node));
            }
        }
    }

    fn reserve_json_work_in(
        &self,
        context: &JsonOperationContext,
        kind: JsonInputKind,
    ) -> Result<JsonReservation> {
        let mut work = context.reservation();
        let mut parents = Vec::<std::slice::Iter<'_, Self>>::new();
        let mut next = Some(self);
        while let Some(node) = next {
            match node {
                Self::And(nodes) => {
                    work.grow(
                        nodes.capacity().saturating_mul(std::mem::size_of::<Self>()),
                        kind,
                    )?;
                    work.reserve_vec(&mut parents, 1, kind)?;
                    parents.push(nodes.iter());
                }
                Self::In(field, values) => {
                    work.grow(
                        field
                            .capacity()
                            .saturating_add(retained_array_bytes(values)),
                        kind,
                    )?;
                }
                Self::Eq(field, value)
                | Self::Ne(field, value)
                | Self::Gt(field, value)
                | Self::Gte(field, value)
                | Self::Lt(field, value)
                | Self::Lte(field, value)
                | Self::Contains(field, value) => {
                    let bytes = match value {
                        JsonValue::Array(values) => retained_array_bytes(values),
                        value => retained_scalar_bytes(value),
                    };
                    work.grow(field.capacity().saturating_add(bytes), kind)?;
                }
            }
            next = loop {
                let Some(parent) = parents.last_mut() else {
                    break None;
                };
                if let Some(node) = parent.next() {
                    break Some(node);
                }
                parents.pop();
            };
        }
        let bytes = parents.capacity() * std::mem::size_of::<std::slice::Iter<'_, Self>>();
        drop(parents);
        work.release(bytes);
        Ok(work)
    }

    #[cfg(test)]
    pub fn evaluate_for_chunk(
        &self,
        catalog: &Catalog,
        chunk: &Chunk,
        expression: JsonValue,
    ) -> FilterExplain {
        let eval = self.evaluate_lookup(&|field| resolve_field(catalog, chunk, field));
        FilterExplain {
            matched: eval.matched,
            expression,
            reasons: eval.reasons,
        }
    }

    pub(crate) fn matches_for_chunk(&self, catalog: &Catalog, chunk: &Chunk) -> bool {
        self.matches_lookup(&|field| resolve_field(catalog, chunk, field))
    }

    fn matches_lookup<'a>(&self, lookup: &impl Fn(&str) -> Option<&'a JsonValue>) -> bool {
        let field = match self {
            Self::And(nodes) => return nodes.iter().all(|node| node.matches_lookup(lookup)),
            Self::Eq(field, _)
            | Self::Ne(field, _)
            | Self::Gt(field, _)
            | Self::Gte(field, _)
            | Self::Lt(field, _)
            | Self::Lte(field, _)
            | Self::In(field, _)
            | Self::Contains(field, _) => field,
        };
        self.matches_value(lookup(field))
    }

    fn matches_value(&self, actual: Option<&JsonValue>) -> bool {
        match self {
            Self::Eq(_, expected) => actual == Some(expected),
            Self::Ne(_, expected) => actual != Some(expected),
            Self::Gt(_, expected) => {
                actual.and_then(|actual| compare_values(actual, expected))
                    == Some(Ordering::Greater)
            }
            Self::Gte(_, expected) => matches!(
                actual.and_then(|actual| compare_values(actual, expected)),
                Some(Ordering::Greater | Ordering::Equal)
            ),
            Self::Lt(_, expected) => {
                actual.and_then(|actual| compare_values(actual, expected)) == Some(Ordering::Less)
            }
            Self::Lte(_, expected) => matches!(
                actual.and_then(|actual| compare_values(actual, expected)),
                Some(Ordering::Less | Ordering::Equal)
            ),
            Self::In(_, values) => actual.is_some_and(|actual| values.contains(actual)),
            Self::Contains(_, needle) => match actual {
                Some(JsonValue::Array(values)) => values.iter().any(|value| value == needle),
                _ => false,
            },
            Self::And(_) => unreachable!(),
        }
    }

    #[cfg(test)]
    fn evaluate_lookup<'a>(&self, lookup: &impl Fn(&str) -> Option<&'a JsonValue>) -> Eval {
        match self {
            FilterExpr::Eq(field, expected) => {
                let actual = lookup(field);
                let matched = self.matches_value(actual);
                reason(
                    matched,
                    || format!("{field} == {}", label(expected)),
                    || {
                        format!(
                            "{field} == {} failed: actual {}",
                            label(expected),
                            actual.map(label).unwrap_or_else(|| "missing".into())
                        )
                    },
                )
            }
            FilterExpr::Ne(field, expected) => {
                let matched = self.matches_value(lookup(field));
                reason(
                    matched,
                    || format!("{field} != {}", label(expected)),
                    || format!("{field} != {} failed", label(expected)),
                )
            }
            FilterExpr::Gt(field, expected)
            | FilterExpr::Gte(field, expected)
            | FilterExpr::Lt(field, expected)
            | FilterExpr::Lte(field, expected) => {
                let actual = lookup(field);
                let op = match self {
                    Self::Gt(_, _) => ">",
                    Self::Gte(_, _) => ">=",
                    Self::Lt(_, _) => "<",
                    Self::Lte(_, _) => "<=",
                    _ => unreachable!(),
                };
                compare_reason(actual, field, expected, op, self.matches_value(actual))
            }
            FilterExpr::In(field, values) => {
                let actual = lookup(field);
                let matched = self.matches_value(actual);
                reason(
                    matched,
                    || {
                        format!(
                            "{field} in [{}]",
                            values.iter().map(label).collect::<Vec<_>>().join(", ")
                        )
                    },
                    || {
                        format!(
                            "{field} in [...] failed: actual {}",
                            actual.map(label).unwrap_or_else(|| "missing".into())
                        )
                    },
                )
            }
            FilterExpr::Contains(field, needle) => {
                let actual = lookup(field);
                let matched = self.matches_value(actual);
                reason(
                    matched,
                    || format!("{field} contains {}", label(needle)),
                    || {
                        format!(
                            "{field} contains {} failed: actual {}",
                            label(needle),
                            actual.map(label).unwrap_or_else(|| "missing".into())
                        )
                    },
                )
            }
            FilterExpr::And(nodes) => {
                let mut matched = true;
                let mut reasons = Vec::new();
                for node in nodes {
                    let eval = node.evaluate_lookup(lookup);
                    matched &= eval.matched;
                    reasons.extend(eval.reasons);
                }
                Eval { matched, reasons }
            }
        }
    }
}

impl FilterExplain {
    pub fn matched(expression: JsonValue) -> Self {
        Self {
            matched: true,
            expression,
            reasons: Vec::new(),
        }
    }

    pub(crate) fn into_json_retained(
        self,
        mut work: JsonReservation,
    ) -> Result<(JsonValue, JsonReservation)> {
        let old_capacity = self.reasons.capacity() * std::mem::size_of::<String>();
        let mut reasons = Vec::new();
        work.reserve_vec(&mut reasons, self.reasons.len(), JsonInputKind::Filter)?;
        for reason in self.reasons {
            reasons.push(JsonValue::String(reason));
        }
        work.release(old_capacity);
        let value =
            crate::json::JsonBuilder::budgeted(&mut work, JsonInputKind::Filter).object([
                ("matched", JsonValue::Bool(self.matched)),
                ("expression", self.expression),
                ("reasons", JsonValue::Array(reasons)),
            ])?;
        Ok((value, work))
    }
}

impl MetadataBoostConfig {
    pub(crate) fn reserve_json_work_in(
        &self,
        context: &JsonOperationContext,
    ) -> Result<JsonReservation> {
        let kind = JsonInputKind::MetadataBoost;
        let mut work = context.reservation();
        work.grow(
            self.rules
                .capacity()
                .saturating_mul(std::mem::size_of::<MetadataBoostRule>())
                .saturating_add(self.hash.capacity()),
            kind,
        )?;
        for rule in &self.rules {
            work.merge(rule.when.reserve_json_work_in(context, kind)?);
        }
        Ok(work)
    }

    pub fn parse(value: JsonValue) -> Result<Self> {
        let context = JsonOperationContext::standalone();
        let (_input, _) = value
            .guard_in(JsonInputKind::MetadataBoost, &context)
            .map_err(|err| {
                if crate::json::is_limit_error(&err)
                    || matches!(err.code(), ErrorCode::IoError | ErrorCode::PermissionDenied)
                {
                    return err.with_detail("field", "metadata_boost");
                }
                Error::new(ErrorCode::InvalidMetadataBoost, err.message())
                    .with_detail("field", "metadata_boost")
                    .with_detail("reason", "invalid_json")
                    .with_detail("metadata_boost_reason", "invalid_json")
            })?;
        Self::parse_validated_in(&value, &context).map(|(value, _)| value)
    }

    pub(crate) fn parse_validated_in(
        value: &JsonValue,
        context: &JsonOperationContext,
    ) -> Result<(Self, JsonReservation)> {
        let mut work = context.reservation();
        let obj = value.as_object().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidMetadataBoost,
                "metadata boost config must be a JSON object",
            )
            .with_detail("field", "metadata_boost")
            .with_detail("metadata_boost_reason", "invalid_type")
            .with_detail("expected", "object")
        })?;
        for key in obj.keys() {
            if !matches!(
                key.as_str(),
                "schema_version" | "rules" | "combine" | "max_boost"
            ) {
                return Err(Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    format!("metadata boost config contains unsupported field {key}"),
                )
                .with_detail("field", format!("metadata_boost.{key}"))
                .with_detail("metadata_boost_reason", "unsupported_field")
                .with_detail("reason", "unsupported_field"));
            }
        }
        match obj.get("schema_version").and_then(JsonValue::as_str) {
            Some("0.1") => {}
            _ => {
                return Err(Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    "metadata boost schema_version must be 0.1",
                )
                .with_detail("field", "metadata_boost.schema_version")
                .with_detail(
                    "metadata_boost_reason",
                    if obj.contains_key("schema_version") {
                        "unsupported_version"
                    } else {
                        "missing_required_field"
                    },
                )
                .with_detail("expected", "0.1"));
            }
        }
        match obj.get("combine").and_then(JsonValue::as_str) {
            Some("sum") => {}
            _ => {
                return Err(Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    "metadata boost combine must be sum",
                )
                .with_detail("field", "metadata_boost.combine")
                .with_detail(
                    "metadata_boost_reason",
                    if obj.contains_key("combine") {
                        "unsupported_combine"
                    } else {
                        "missing_required_field"
                    },
                )
                .with_detail("expected", "sum"));
            }
        }
        let max_boost = non_negative_finite_number(obj, "max_boost", "metadata_boost.max_boost")?;
        let rules_json = obj
            .get("rules")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    "metadata boost rules must be an array",
                )
                .with_detail("field", "metadata_boost.rules")
                .with_detail(
                    "metadata_boost_reason",
                    if obj.contains_key("rules") {
                        "invalid_type"
                    } else {
                        "missing_required_field"
                    },
                )
                .with_detail("expected", "array")
            })?;
        if rules_json.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidMetadataBoost,
                "metadata boost rules must not be empty",
            )
            .with_detail("field", "metadata_boost.rules")
            .with_detail("metadata_boost_reason", "empty")
            .with_detail("reason", "empty"));
        }
        let mut rules = Vec::new();
        work.reserve_vec(&mut rules, rules_json.len(), JsonInputKind::MetadataBoost)?;
        for (index, rule_json) in rules_json.iter().enumerate() {
            let rule_obj = rule_json.as_object().ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    format!("metadata boost rules[{index}] must be an object"),
                )
                .with_detail("field", format!("metadata_boost.rules[{index}]"))
                .with_detail("metadata_boost_reason", "invalid_type")
                .with_detail("expected", "object")
            })?;
            for key in rule_obj.keys() {
                if !matches!(key.as_str(), "when" | "boost") {
                    return Err(Error::new(
                        ErrorCode::InvalidMetadataBoost,
                        format!("metadata boost rules[{index}] contains unsupported field {key}"),
                    )
                    .with_detail("field", format!("metadata_boost.rules[{index}].{key}"))
                    .with_detail("metadata_boost_reason", "unsupported_field")
                    .with_detail("reason", "unsupported_field"));
                }
            }
            let when_json = rule_obj.get("when").ok_or_else(|| {
                Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    format!("metadata boost rules[{index}].when is required"),
                )
                .with_detail("field", format!("metadata_boost.rules[{index}].when"))
                .with_detail("metadata_boost_reason", "missing_required_field")
                .with_detail("expected", "metadata_filter")
            })?;
            let when =
                FilterExpr::parse_with_work(when_json, &mut work, JsonInputKind::MetadataBoost)
                    .map_err(|err| {
                        if crate::json::is_limit_error(&err)
                            || matches!(
                                err.code(),
                                ErrorCode::IoError | ErrorCode::PermissionDenied
                            )
                        {
                            return err.with_detail("field", "metadata_boost");
                        }
                        Error::new(
                            ErrorCode::InvalidMetadataBoost,
                            format!(
                                "metadata boost rules[{index}].when is invalid: {}",
                                err.message()
                            ),
                        )
                        .with_detail("field", format!("metadata_boost.rules[{index}].when"))
                        .with_detail("metadata_boost_reason", "invalid_filter")
                        .with_detail("reason", "invalid_filter")
                    })?;
            let Some(when) = when else {
                return Err(Error::new(
                    ErrorCode::InvalidMetadataBoost,
                    format!("metadata boost rules[{index}].when must not be empty"),
                )
                .with_detail("field", format!("metadata_boost.rules[{index}].when"))
                .with_detail("metadata_boost_reason", "empty_filter")
                .with_detail("reason", "empty_filter"));
            };
            let boost = non_negative_finite_number(
                rule_obj,
                "boost",
                format!("metadata_boost.rules[{index}].boost"),
            )?;
            rules.push(MetadataBoostRule { when, boost });
        }
        // All fields are required and validated above; the input is already the normalized shape.
        let (_, hash, hash_work) =
            value.canonical_digest_validated_in(JsonInputKind::MetadataBoost, context)?;
        work.merge(hash_work);
        Ok((
            Self {
                rules,
                max_boost,
                hash,
            },
            work,
        ))
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn hash(&self) -> &str {
        &self.hash
    }

    pub fn score_for_chunk(&self, catalog: &Catalog, chunk: &Chunk) -> MetadataBoostScore {
        let mut total = 0.0;
        let mut matched_rule_count = 0;
        for rule in &self.rules {
            if rule.when.matches_for_chunk(catalog, chunk) {
                matched_rule_count += 1;
                total += rule.boost;
            }
        }
        MetadataBoostScore {
            boost: total.min(self.max_boost),
            matched_rule_count,
        }
    }
}

#[derive(Clone, Debug)]
struct Eval {
    matched: bool,
    reasons: Vec<String>,
}

// Compiled boost literals are immutable scalars or flat scalar arrays.
fn retained_array_bytes(values: &Vec<JsonValue>) -> usize {
    values.iter().fold(
        values
            .capacity()
            .saturating_mul(std::mem::size_of::<JsonValue>()),
        |bytes, value| bytes.saturating_add(retained_scalar_bytes(value)),
    )
}

fn retained_scalar_bytes(value: &JsonValue) -> usize {
    match value {
        JsonValue::String(value) => value.capacity(),
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) => 0,
        _ => unreachable!("compiled filter literals were validated"),
    }
}

fn filter_object(value: &JsonValue) -> Result<Option<&JsonObject>> {
    if value.is_null() {
        return Ok(None);
    }
    let obj = value.as_object().ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidMetadataFilter,
            "filter must be a JSON object",
        )
        .with_detail("field", "filter")
        .with_detail("expected", "object")
    })?;
    Ok((!obj.is_empty()).then_some(obj))
}

fn copy_filter_literal(
    value: &JsonValue,
    work: &mut JsonReservation,
    kind: JsonInputKind,
) -> Result<JsonValue> {
    fn scalar(
        value: &JsonValue,
        work: &mut JsonReservation,
        kind: JsonInputKind,
    ) -> Result<JsonValue> {
        Ok(match value {
            JsonValue::String(value) => JsonValue::String(work.copy_string(value, kind)?),
            JsonValue::Null => JsonValue::Null,
            JsonValue::Bool(value) => JsonValue::Bool(*value),
            JsonValue::Number(value) => JsonValue::Number(*value),
            _ => unreachable!("filter literals were validated before copying"),
        })
    }
    if let JsonValue::Array(values) = value {
        let mut copy = Vec::new();
        work.reserve_vec(&mut copy, values.len(), kind)?;
        for value in values {
            copy.push(scalar(value, work, kind)?);
        }
        Ok(JsonValue::Array(copy))
    } else {
        scalar(value, work, kind)
    }
}

fn parse_field_condition(
    field: &str,
    condition: &JsonValue,
    work: &mut JsonReservation,
    kind: JsonInputKind,
) -> Result<FilterExpr> {
    let detail_field = format!("filter.{field}");
    let Some(obj) = condition.as_object() else {
        validate_filter_literal(condition, "$eq", &detail_field)?;
        return Ok(FilterExpr::Eq(
            work.copy_string(field, kind)?,
            copy_filter_literal(condition, work, kind)?,
        ));
    };
    if obj.len() != 1 {
        return Err(Error::new(
            ErrorCode::InvalidMetadataFilter,
            format!("field {field} must have exactly one operator"),
        )
        .with_detail("field", detail_field)
        .with_detail("reason", "operator_count")
        .with_detail("actual_count", obj.len()));
    }
    let (op, value) = obj.iter().next().expect("len checked");
    match op.as_str() {
        "$eq" | "$ne" => {
            validate_filter_literal(value, op, &format!("{detail_field}.{op}"))?;
        }
        "$gt" | "$gte" | "$lt" | "$lte" => {
            validate_range_literal(value, op, &format!("{detail_field}.{op}"))?;
        }
        "$in" => {
            let values = value.as_array().ok_or_else(|| {
                Error::new(ErrorCode::InvalidMetadataFilter, "$in requires an array")
                    .with_detail("field", format!("{detail_field}.$in"))
                    .with_detail("expected", "array")
            })?;
            for item in values {
                validate_scalar_literal(item, "$in", &format!("{detail_field}.$in[]"))?;
            }
        }
        "$contains" => {
            validate_scalar_literal(value, "$contains", &format!("{detail_field}.$contains"))?;
        }
        other => {
            return Err(Error::new(
                ErrorCode::InvalidMetadataFilter,
                format!("unsupported filter operator {other}"),
            )
            .with_detail("field", detail_field)
            .with_detail("reason", "unsupported_operator")
            .with_detail("operator", other))
        }
    }
    let field = work.copy_string(field, kind)?;
    let mut value = copy_filter_literal(value, work, kind)?;
    Ok(match op.as_str() {
        "$eq" => FilterExpr::Eq(field, value),
        "$ne" => FilterExpr::Ne(field, value),
        "$gt" => FilterExpr::Gt(field, value),
        "$gte" => FilterExpr::Gte(field, value),
        "$lt" => FilterExpr::Lt(field, value),
        "$lte" => FilterExpr::Lte(field, value),
        "$contains" => FilterExpr::Contains(field, value),
        "$in" => {
            let JsonValue::Array(values) = &mut value else {
                unreachable!()
            };
            FilterExpr::In(field, std::mem::take(values))
        }
        _ => unreachable!(),
    })
}

fn validate_filter_literal(value: &JsonValue, op: &str, detail_field: &str) -> Result<()> {
    match value {
        JsonValue::Array(values) => {
            for item in values {
                validate_scalar_literal(item, op, detail_field)?;
            }
            Ok(())
        }
        _ => validate_scalar_literal(value, op, detail_field),
    }
}

fn validate_scalar_literal(value: &JsonValue, op: &str, detail_field: &str) -> Result<()> {
    match value {
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {
            Ok(())
        }
        JsonValue::Array(_) | JsonValue::Object(_) => Err(Error::new(
            ErrorCode::InvalidMetadataFilter,
            format!("{op} only supports scalar JSON values"),
        )
        .with_detail("field", detail_field)
        .with_detail("reason", "scalar_only")),
    }
}

fn validate_range_literal(value: &JsonValue, op: &str, detail_field: &str) -> Result<()> {
    match value {
        JsonValue::Number(_) | JsonValue::String(_) => Ok(()),
        _ => Err(Error::new(
            ErrorCode::InvalidMetadataFilter,
            format!("{op} requires a number or string value"),
        )
        .with_detail("field", detail_field)
        .with_detail("expected", "number_or_string")),
    }
}

fn validate_filter_field(field: &str) -> Result<()> {
    let Some((prefix, key)) = field.split_once('.') else {
        return Err(Error::new(
            ErrorCode::InvalidMetadataFilter,
            format!("filter field {field} must use doc.<key> or chunk.<key>"),
        )
        .with_detail("field", format!("filter.{field}"))
        .with_detail("reason", "missing_scope_prefix")
        .with_detail("expected", "doc.<key>|chunk.<key>"));
    };
    if !matches!(prefix, "doc" | "chunk") || key.is_empty() || key.contains('.') {
        return Err(Error::new(
            ErrorCode::InvalidMetadataFilter,
            format!("filter field {field} must be doc.<top_level_key> or chunk.<top_level_key>"),
        )
        .with_detail("field", format!("filter.{field}"))
        .with_detail("reason", "invalid_scope_or_nested_path")
        .with_detail("expected", "doc.<top_level_key>|chunk.<top_level_key>"));
    }
    Ok(())
}

#[cfg(test)]
fn compare_reason(
    actual: Option<&JsonValue>,
    field: &str,
    expected: &JsonValue,
    op: &str,
    matched: bool,
) -> Eval {
    reason(
        matched,
        || format!("{field} {op} {}", label(expected)),
        || {
            format!(
                "{field} {op} {} failed: actual {}",
                label(expected),
                actual.map(label).unwrap_or_else(|| "missing".into())
            )
        },
    )
}

fn compare_values(actual: &JsonValue, expected: &JsonValue) -> Option<Ordering> {
    match (actual, expected) {
        (JsonValue::Number(a), JsonValue::Number(b)) => a.partial_cmp(b),
        (JsonValue::String(a), JsonValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn non_negative_finite_number(
    obj: &JsonObject,
    field: &str,
    detail_field: impl ToString,
) -> Result<f64> {
    let detail_field = detail_field.to_string();
    let value = obj.get(field).and_then(JsonValue::as_f64).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidMetadataBoost,
            format!("metadata boost {field} must be a number"),
        )
        .with_detail("field", detail_field.clone())
        .with_detail(
            "metadata_boost_reason",
            if obj.contains_key(field) {
                "invalid_type"
            } else {
                "missing_required_field"
            },
        )
        .with_detail("expected", "number")
    })?;
    if !value.is_finite() || value < 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidMetadataBoost,
            format!("metadata boost {field} must be finite and non-negative"),
        )
        .with_detail("field", detail_field)
        .with_detail("reason", "finite_non_negative")
        .with_detail("metadata_boost_reason", "finite_non_negative")
        .with_detail("actual", value));
    }
    Ok(value)
}

fn resolve_field<'a>(catalog: &'a Catalog, chunk: &'a Chunk, field: &str) -> Option<&'a JsonValue> {
    let (scope, key) = field.split_once('.')?;
    match scope {
        "doc" => {
            let document = catalog.documents.get(&chunk.document_id)?;
            document.metadata.get(key)
        }
        "chunk" => chunk.metadata.get(key),
        _ => None,
    }
}

#[cfg(test)]
fn reason(matched: bool, ok: impl FnOnce() -> String, fail: impl FnOnce() -> String) -> Eval {
    Eval {
        matched,
        reasons: vec![if matched { ok() } else { fail() }],
    }
}

#[cfg(test)]
fn label(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => {
            if value.fract() == 0.0 {
                format!("{value:.0}")
            } else {
                value.to_string()
            }
        }
        JsonValue::String(value) => value.clone(),
        JsonValue::Array(_) | JsonValue::Object(_) => value.to_compact_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::json::parse;
    use crate::model::Chunk;

    #[test]
    fn boost_guard_native_failure_is_not_wrapped_as_invalid_json() {
        let value = parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"chunk.x":true},"boost":0.1}]}"#).unwrap();
        let retry = value.clone();
        // Include cold lock initialization before the fallible traversal array.
        let (_, setup_allocations) = crate::test_alloc::allocations(|| {
            let context = JsonOperationContext::standalone();
            std::hint::black_box(context.available_workspace_bytes());
            std::hint::black_box(context)
        });
        let error = crate::test_alloc::fail_nth_allocation_once(setup_allocations + 1, || {
            MetadataBoostConfig::parse(value)
        })
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(error_detail_value(&error, "field"), Some("metadata_boost"));
        assert_eq!(error_detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(error_detail_value(&error, "json_limit_reason"), None);
        assert!(!crate::json::is_memory_budget_error(&error));
        assert_eq!(MetadataBoostConfig::parse(retry).unwrap().rule_count(), 1);
    }

    #[test]
    fn boost_filter_native_copy_failure_preserves_source_credit_and_system_error() {
        let payload = "x".repeat(131_071);
        let value = parse(&format!("{{\"schema_version\":\"0.1\",\"combine\":\"sum\",\"max_boost\":1,\"rules\":[{{\"when\":{{\"chunk.x\":\"{payload}\"}},\"boost\":0.1}}]}}" )).unwrap();
        let context = JsonOperationContext::standalone();
        let capacity = context.available_workspace_bytes();
        let (input_work, _) = value
            .guard_in(JsonInputKind::MetadataBoost, &context)
            .unwrap();
        let baseline = context.available_workspace_bytes();
        let error = crate::test_alloc::fail_byte_allocation_once(payload.len(), || {
            MetadataBoostConfig::parse_validated_in(&value, &context)
        })
        .err()
        .unwrap();
        assert_eq!(error.code(), ErrorCode::IoError);
        assert_eq!(error_detail_value(&error, "field"), Some("metadata_boost"));
        assert_eq!(error_detail_value(&error, "metadata_boost_reason"), None);
        assert_eq!(error_detail_value(&error, "json_limit_reason"), None);
        assert_eq!(context.available_workspace_bytes(), baseline);
        let (config, work) = MetadataBoostConfig::parse_validated_in(&value, &context).unwrap();
        assert_eq!(config.rule_count(), 1);
        drop(config);
        drop(work);
        assert_eq!(context.available_workspace_bytes(), baseline);
        drop(value);
        drop(input_work);
        assert_eq!(context.available_workspace_bytes(), capacity);
    }

    #[test]
    fn compiled_boost_adoption_counts_the_retained_representation_once() {
        let value = parse(r#"{"schema_version":"0.1","combine":"sum","max_boost":1,"rules":[{"when":{"$and":[{"chunk.x":{"$in":[null,"a",2]}},{"chunk.y":[1,"text"]},{"chunk.z":"q"}]},"boost":0.1}]}"#).unwrap();
        let context = JsonOperationContext::standalone();
        let (_input, _) = value
            .guard_in(JsonInputKind::MetadataBoost, &context)
            .unwrap();
        let (config, compiled) = MetadataBoostConfig::parse_validated_in(&value, &context).unwrap();
        let adopted = config.reserve_json_work_in(&context).unwrap();
        assert_eq!(adopted.reserved_bytes(), compiled.reserved_bytes());
        let cloned = config.clone();
        let cloned_work = cloned.reserve_json_work_in(&context).unwrap();
        assert!(cloned_work.reserved_bytes() <= adopted.reserved_bytes());
        assert!(cloned_work.reserved_bytes() >= 64);
    }

    #[test]
    fn compiled_filter_and_boost_copies_share_the_input_budget() {
        let filter = || {
            object_from_pairs([(
                "chunk.secret_marker",
                object_from_pairs([("$in", JsonValue::Array(vec![JsonValue::Null; 600_000]))]),
            )])
        };
        let value = filter();
        value.validate_kind(JsonInputKind::Filter).unwrap();
        let error = FilterExpr::parse(value).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataFilter);
        assert_eq!(
            error_detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        let value = object_from_pairs([
            ("schema_version", JsonValue::String("0.1".into())),
            ("combine", JsonValue::String("sum".into())),
            ("max_boost", JsonValue::Number(1.0)),
            (
                "rules",
                JsonValue::Array(vec![object_from_pairs([
                    ("when", filter()),
                    ("boost", JsonValue::Number(0.1)),
                ])]),
            ),
        ]);
        value.validate_kind(JsonInputKind::MetadataBoost).unwrap();
        let error = MetadataBoostConfig::parse(value).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            error_detail_value(&error, "json_limit_reason"),
            Some("work_budget_exceeded")
        );
        assert_eq!(
            error_detail_value(&error, "json_input_kind"),
            Some("metadata_boost")
        );
        assert_eq!(
            error_detail_value(&error, "metadata_boost_reason"),
            Some("json_processing_limit")
        );
        assert!(!error
            .to_json()
            .to_compact_string()
            .contains("secret_marker"));
    }

    #[test]
    fn compiler_preserves_empty_and_nested_and_semantics_without_recursive_parsing() {
        assert!(FilterExpr::parse(JsonValue::Null).unwrap().is_none());
        assert!(FilterExpr::parse(JsonValue::object()).unwrap().is_none());
        let empty = FilterExpr::parse(parse(r#"{"$and":[{},null,{"$and":[]}] }"#).unwrap())
            .unwrap()
            .unwrap();
        assert!(empty.matches_lookup(&|_| None));
        for n in 0..100 {
            let raw = format!(
                r#"{{"$and":[null,{{}},{{"chunk.a":{{"$gte":{n}}}}},{{"$and":[{{"chunk.b":false}},{{"chunk.c":{{"$in":[null,1,"x"]}}}}]}}],"chunk.d":[1,2]}}"#
            );
            let compiled = FilterExpr::parse(parse(&raw).unwrap()).unwrap().unwrap();
            let expected = FilterExpr::And(vec![
                FilterExpr::Gte("chunk.a".into(), JsonValue::Number(n as f64)),
                FilterExpr::And(vec![
                    FilterExpr::Eq("chunk.b".into(), JsonValue::Bool(false)),
                    FilterExpr::In(
                        "chunk.c".into(),
                        vec![
                            JsonValue::Null,
                            JsonValue::Number(1.0),
                            JsonValue::String("x".into()),
                        ],
                    ),
                ]),
                FilterExpr::Eq(
                    "chunk.d".into(),
                    JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]),
                ),
            ]);
            assert_eq!(format!("{compiled:?}"), format!("{expected:?}"));
        }
        let mut value = parse(r#"{"chunk.a":1}"#).unwrap();
        for _ in 0..63 {
            value = object_from_pairs([("$and", JsonValue::Array(vec![value]))]);
        }
        let result = std::thread::Builder::new()
            .stack_size(64 * 1024)
            .spawn(move || FilterExpr::parse(value).unwrap())
            .unwrap()
            .join()
            .unwrap();
        assert!(matches!(result, Some(FilterExpr::Eq(_, _))));
    }

    #[test]
    fn boost_hash_uses_the_same_normalized_values_without_an_extra_json_copy() {
        for n in 0..100 {
            let raw = format!(
                r#"{{"schema_version":"0.1","combine":"sum","max_boost":-0.0,"rules":[{{"when":{{"chunk.label":"value-{n}é\n"}},"boost":1e-1}}]}}"#
            );
            let value = parse(&raw).unwrap();
            let expected = crate::hash::sha256_hex(value.to_compact_string().as_bytes());
            let config = MetadataBoostConfig::parse(value).unwrap();
            assert_eq!(config.hash(), expected);
        }
    }

    #[test]
    fn borrowed_lookup_matches_materialized_metadata_across_one_hundred_fixtures() {
        for seed in 0..100 {
            let mut catalog = Catalog::default();
            let document = crate::model::Document {
                document_id: "doc".into(), collection_id: "col".into(),
                external_id: format!("real-doc-{seed}"), title: None,
                source_uri: (seed % 2 == 0).then(|| format!("real-uri-{seed}")),
                metadata: parse(&format!(r#"{{"n":{seed},"external_id":"shadow-doc","source_uri":"fallback","null":null}}"#)).unwrap(),
                created_at: "1970-01-01T00:00:00Z".into(), updated_at: "1970-01-01T00:00:00Z".into(),
            };
            let chunk = Chunk {
                chunk_id: "chunk".into(),
                document_id: "doc".into(),
                collection_id: "col".into(),
                chunk_external_id: format!("real-chunk-{seed}"),
                text: "text".into(),
                text_hash: "hash".into(),
                ordinal: 0,
                token_count: 1,
                metadata: parse(&format!(
                    r#"{{"n":{},"chunk_external_id":"shadow-chunk","tags":["tag",null,1]}}"#,
                    99 - seed
                ))
                .unwrap(),
                embedding_id: None,
                source_span: None,
                created_at: "1970-01-01T00:00:00Z".into(),
            };
            let materialized = object_from_pairs([
                ("doc", document.metadata.clone()),
                ("chunk", chunk.metadata.clone()),
            ]);
            catalog.documents.insert("doc".into(), Box::new(document));
            let lookup = |field: &str| {
                let (scope, key) = field.split_once('.')?;
                materialized.get(scope)?.get(key)
            };
            for field in [
                "doc.n",
                "chunk.n",
                "doc.external_id",
                "doc.source_uri",
                "chunk.chunk_external_id",
                "doc.missing",
                "doc.null",
                "chunk.tags",
            ] {
                for filter in [
                    FilterExpr::Eq(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::Ne(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::Gt(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::Gte(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::Lt(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::Lte(field.into(), JsonValue::Number(seed as f64)),
                    FilterExpr::In(
                        field.into(),
                        vec![
                            JsonValue::Null,
                            JsonValue::String(format!("real-doc-{seed}")),
                        ],
                    ),
                    FilterExpr::Contains(field.into(), JsonValue::String("tag".into())),
                ] {
                    let reference = filter.evaluate_lookup(&lookup);
                    assert_eq!(
                        filter.matches_for_chunk(&catalog, &chunk),
                        reference.matched,
                        "seed={seed}, field={field}"
                    );
                    let explain = filter.evaluate_for_chunk(&catalog, &chunk, JsonValue::object());
                    assert_eq!(explain.matched, reference.matched);
                    assert_eq!(explain.reasons, reference.reasons);
                    let context = JsonOperationContext::standalone();
                    let (explain, work) = filter
                        .evaluate_for_chunk_in(&catalog, &chunk, JsonValue::object(), &context)
                        .unwrap();
                    assert_eq!(explain.matched, reference.matched);
                    assert_eq!(explain.reasons, reference.reasons);
                    let (json, work) = explain.into_json_retained(work).unwrap();
                    let measured = json
                        .reserve_retained_in(JsonInputKind::Filter, &context)
                        .unwrap();
                    assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
                }
            }
            let original = catalog
                .documents
                .get("doc")
                .unwrap()
                .metadata
                .get("n")
                .unwrap();
            assert!(std::ptr::eq(
                original,
                resolve_field(&catalog, &chunk, "doc.n").unwrap()
            ));
        }
    }

    #[test]
    fn boolean_matching_short_circuits_but_explain_keeps_every_reason() {
        let count = std::cell::Cell::new(0);
        let actual = JsonValue::Number(0.0);
        let filter = FilterExpr::And(vec![
            FilterExpr::Eq("doc.first".into(), JsonValue::Number(1.0)),
            FilterExpr::Eq("doc.second".into(), JsonValue::Number(0.0)),
        ]);
        let lookup = |_: &str| {
            count.set(count.get() + 1);
            Some(&actual)
        };
        assert!(!filter.matches_lookup(&lookup));
        assert_eq!(count.get(), 1);
        count.set(0);
        let explain = filter.evaluate_lookup(&lookup);
        assert!(!explain.matched);
        assert_eq!(count.get(), 2);
        assert_eq!(
            explain.reasons,
            ["doc.first == 1 failed: actual 0", "doc.second == 0"]
        );
    }

    #[test]
    fn successful_contains_does_not_format_the_unused_failure_reason() {
        let eval = reason(
            true,
            || "matched".into(),
            || panic!("unused diagnostic must not allocate"),
        );
        assert_eq!(eval.reasons, ["matched"]);
        let eval = reason(
            false,
            || panic!("unused success must not allocate"),
            || "failed".into(),
        );
        assert_eq!(eval.reasons, ["failed"]);
    }

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn parses_and_evaluates_doc_chunk_paths() {
        let filter =
            FilterExpr::parse(parse(r#"{"chunk.lang":"rust","chunk.stars":{"$gte":10}}"#).unwrap())
                .unwrap()
                .unwrap();
        let mut catalog = Catalog::default();
        let chunk = Chunk {
            chunk_id: "c".into(),
            document_id: "d".into(),
            collection_id: "col".into(),
            chunk_external_id: "cx".into(),
            text: "fn main".into(),
            text_hash: "h".into(),
            ordinal: 0,
            token_count: 2,
            metadata: parse(r#"{"lang":"rust","stars":12}"#).unwrap(),
            embedding_id: None,
            source_span: None,
            created_at: "1970-01-01T00:00:00Z".into(),
        };
        catalog
            .chunks
            .insert(chunk.chunk_id.clone(), Box::new(chunk.clone()));
        let explain = filter.evaluate_for_chunk(&catalog, &chunk, JsonValue::object());
        assert!(explain.matched);
    }

    #[test]
    fn rejects_unstable_filter_paths_and_string_contains() {
        assert!(FilterExpr::parse(parse(r#"{"stars":{"$gte":10}}"#).unwrap()).is_err());
        assert!(FilterExpr::parse(parse(r#"{"chunk.nested.value":1}"#).unwrap()).is_err());

        let filter = FilterExpr::parse(parse(r#"{"chunk.lang":{"$contains":"ru"}}"#).unwrap())
            .unwrap()
            .unwrap();
        let catalog = Catalog::default();
        let chunk = Chunk {
            chunk_id: "c".into(),
            document_id: "d".into(),
            collection_id: "col".into(),
            chunk_external_id: "cx".into(),
            text: "fn main".into(),
            text_hash: "h".into(),
            ordinal: 0,
            token_count: 2,
            metadata: parse(r#"{"lang":"rust"}"#).unwrap(),
            embedding_id: None,
            source_span: None,
            created_at: "1970-01-01T00:00:00Z".into(),
        };
        assert!(
            !filter
                .evaluate_for_chunk(&catalog, &chunk, JsonValue::object())
                .matched
        );
    }

    #[test]
    fn rejects_object_filter_operands() {
        assert!(
            FilterExpr::parse(parse(r#"{"chunk.meta":{"$eq":{"nested":true}}}"#).unwrap()).is_err()
        );
        assert!(FilterExpr::parse(
            parse(r#"{"chunk.tags":{"$contains":{"name":"rust"}}}"#).unwrap()
        )
        .is_err());
        assert!(FilterExpr::parse(
            parse(r#"{"chunk.tags":{"$in":["rust",{"name":"db"}]}}"#).unwrap()
        )
        .is_err());
        assert!(FilterExpr::parse(parse(r#"{"chunk.score":{"$gte":true}}"#).unwrap()).is_err());
    }

    #[test]
    fn metadata_filter_errors_include_machine_readable_detail() {
        let non_object = FilterExpr::parse(JsonValue::String("bad".into())).unwrap_err();
        assert_eq!(non_object.code(), ErrorCode::InvalidMetadataFilter);
        assert_eq!(error_detail_value(&non_object, "field"), Some("filter"));
        assert_eq!(error_detail_value(&non_object, "expected"), Some("object"));

        let missing_prefix =
            FilterExpr::parse(parse(r#"{"stars":{"$gte":10}}"#).unwrap()).unwrap_err();
        assert_eq!(
            error_detail_value(&missing_prefix, "field"),
            Some("filter.stars")
        );
        assert_eq!(
            error_detail_value(&missing_prefix, "reason"),
            Some("missing_scope_prefix")
        );

        let nested_path =
            FilterExpr::parse(parse(r#"{"chunk.nested.value":1}"#).unwrap()).unwrap_err();
        assert_eq!(
            error_detail_value(&nested_path, "field"),
            Some("filter.chunk.nested.value")
        );
        assert_eq!(
            error_detail_value(&nested_path, "reason"),
            Some("invalid_scope_or_nested_path")
        );

        let object_operand =
            FilterExpr::parse(parse(r#"{"chunk.meta":{"$eq":{"nested":true}}}"#).unwrap())
                .unwrap_err();
        assert_eq!(
            error_detail_value(&object_operand, "field"),
            Some("filter.chunk.meta.$eq")
        );
        assert_eq!(
            error_detail_value(&object_operand, "reason"),
            Some("scalar_only")
        );

        let range_type =
            FilterExpr::parse(parse(r#"{"chunk.score":{"$gte":true}}"#).unwrap()).unwrap_err();
        assert_eq!(
            error_detail_value(&range_type, "field"),
            Some("filter.chunk.score.$gte")
        );
        assert_eq!(
            error_detail_value(&range_type, "expected"),
            Some("number_or_string")
        );

        let unsupported_operator =
            FilterExpr::parse(parse(r#"{"chunk.tags":{"$between":[1,2]}}"#).unwrap()).unwrap_err();
        assert_eq!(
            error_detail_value(&unsupported_operator, "field"),
            Some("filter.chunk.tags")
        );
        assert_eq!(
            error_detail_value(&unsupported_operator, "reason"),
            Some("unsupported_operator")
        );
        assert_eq!(
            error_detail_value(&unsupported_operator, "operator"),
            Some("$between")
        );
    }

    #[test]
    fn evaluates_scalar_array_metadata_filters() {
        let filter = FilterExpr::parse(
            parse(
                r#"{
                    "chunk.tags":{"$contains":"storage"},
                    "chunk.score":{"$gt":10.5},
                    "chunk.updated_at":{"$lte":"2026-05-31T00:00:00Z"}
                }"#,
            )
            .unwrap(),
        )
        .unwrap()
        .unwrap();
        let catalog = Catalog::default();
        let chunk = Chunk {
            chunk_id: "c".into(),
            document_id: "d".into(),
            collection_id: "col".into(),
            chunk_external_id: "cx".into(),
            text: "fn main".into(),
            text_hash: "h".into(),
            ordinal: 0,
            token_count: 2,
            metadata: parse(
                r#"{"tags":["storage","rust"],"score":11,"updated_at":"2026-05-30T23:00:00Z"}"#,
            )
            .unwrap(),
            embedding_id: None,
            source_span: None,
            created_at: "1970-01-01T00:00:00Z".into(),
        };
        assert!(
            filter
                .evaluate_for_chunk(&catalog, &chunk, JsonValue::object())
                .matched
        );
    }

    #[test]
    fn metadata_boost_config_validates_and_scores_candidate_metadata() {
        let config = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[
                        {"when":{"chunk.importance":{"$gte":0.8}},"boost":0.08},
                        {"when":{"doc.kind":"guide"},"boost":0.08}
                    ],
                    "combine":"sum",
                    "max_boost":0.1
                }"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(config.rule_count(), 2);
        assert_eq!(config.hash().len(), 64);

        let mut catalog = Catalog::default();
        catalog.documents.insert(
            "d".into(),
            Box::new(crate::model::Document {
                document_id: "d".into(),
                collection_id: "col".into(),
                external_id: "doc".into(),
                title: None,
                source_uri: None,
                metadata: parse(r#"{"kind":"guide"}"#).unwrap(),
                created_at: "1970-01-01T00:00:00Z".into(),
                updated_at: "1970-01-01T00:00:00Z".into(),
            }),
        );
        let chunk = Chunk {
            chunk_id: "c".into(),
            document_id: "d".into(),
            collection_id: "col".into(),
            chunk_external_id: "cx".into(),
            text: "fn main".into(),
            text_hash: "h".into(),
            ordinal: 0,
            token_count: 2,
            metadata: parse(r#"{"importance":0.9}"#).unwrap(),
            embedding_id: None,
            source_span: None,
            created_at: "1970-01-01T00:00:00Z".into(),
        };
        let score = config.score_for_chunk(&catalog, &chunk);
        assert_eq!(score.matched_rule_count, 2);
        assert_eq!(score.boost, 0.1);

        let err = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{},"boost":0.1}],
                    "combine":"sum",
                    "max_boost":0.1
                }"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            error_detail_value(&err, "field"),
            Some("metadata_boost.rules[0].when")
        );
        assert_eq!(error_detail_value(&err, "reason"), Some("empty_filter"));
    }

    #[test]
    fn metadata_boost_errors_include_machine_readable_detail() {
        let unknown = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{"chunk.lang":"rust"},"boost":0.1}],
                    "combine":"sum",
                    "max_boost":0.1,
                    "debug":true
                }"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(unknown.code(), ErrorCode::InvalidMetadataBoost);
        assert_eq!(
            error_detail_value(&unknown, "field"),
            Some("metadata_boost.debug")
        );
        assert_eq!(
            error_detail_value(&unknown, "reason"),
            Some("unsupported_field")
        );

        let missing_rule_boost = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{"chunk.lang":"rust"}}],
                    "combine":"sum",
                    "max_boost":0.1
                }"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            error_detail_value(&missing_rule_boost, "field"),
            Some("metadata_boost.rules[0].boost")
        );
        assert_eq!(
            error_detail_value(&missing_rule_boost, "expected"),
            Some("number")
        );

        let invalid_rule_filter = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{"chunk.lang":{"$eq":{"nested":true}}},"boost":0.1}],
                    "combine":"sum",
                    "max_boost":0.1
                }"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            error_detail_value(&invalid_rule_filter, "field"),
            Some("metadata_boost.rules[0].when")
        );
        assert_eq!(
            error_detail_value(&invalid_rule_filter, "reason"),
            Some("invalid_filter")
        );
    }

    #[test]
    fn metadata_boost_reason_covers_non_finite_rust_inputs() {
        for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in ["max_boost", "boost", "when"] {
                let mut rule = object_from_pairs([
                    ("when", parse(r#"{"chunk.kind":"rust"}"#).unwrap()),
                    ("boost", JsonValue::Number(0.5)),
                ]);
                if field != "max_boost" {
                    rule.as_object_mut()
                        .unwrap()
                        .insert(field.into(), JsonValue::Number(number));
                }
                let input = object_from_pairs([
                    ("schema_version", JsonValue::String("0.1".into())),
                    ("combine", JsonValue::String("sum".into())),
                    (
                        "max_boost",
                        JsonValue::Number(if field == "max_boost" { number } else { 1.0 }),
                    ),
                    ("rules", JsonValue::Array(vec![rule])),
                ]);
                let error = MetadataBoostConfig::parse(input).unwrap_err();
                assert_eq!(error.code(), ErrorCode::InvalidMetadataBoost);
                assert_eq!(
                    error_detail_value(&error, "metadata_boost_reason"),
                    Some("invalid_json")
                );
                assert_eq!(error_detail_value(&error, "field"), Some("metadata_boost"));
                assert!(parse(&error.to_json().to_compact_string()).is_ok());
            }
        }
    }

    #[test]
    fn metadata_boost_hash_is_stable_across_key_order_and_whitespace() {
        let first = MetadataBoostConfig::parse(
            parse(
                r#"{
                    "schema_version":"0.1",
                    "rules":[{"when":{"chunk.importance":{"$gte":0.8}},"boost":0.1}],
                    "combine":"sum",
                    "max_boost":0.2
                }"#,
            )
            .unwrap(),
        )
        .unwrap();
        let reordered = MetadataBoostConfig::parse(
            parse(
                r#"{"max_boost":0.2,"combine":"sum","rules":[{"boost":0.1,"when":{"chunk.importance":{"$gte":0.8}}}],"schema_version":"0.1"}"#,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(first.hash(), reordered.hash());
    }
}
