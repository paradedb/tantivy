//! Writes the synonym component of a segment, and merges it.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use common::TerminatingWrite;

use super::dictionary::{encode_key, groups, GroupRef, SynonymFileWriter, SYNONYM_EXT};
use super::rule::{CompiledRule, SynonymRule};
use crate::error::DataCorruption;
use crate::index::{Segment, SegmentComponent};
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::schema::document::{ErasedDocument, ErasedValue, ReferenceValueLeaf};
use crate::schema::{Field, FieldType, Schema};
use crate::tokenizer::TextAnalyzer;
use crate::{DocId, TantivyError};

pub(crate) struct SynonymPlugin;

pub(crate) fn synonym_rule_field(schema: &Schema) -> crate::Result<Option<Field>> {
    let mut fields = schema
        .fields()
        .filter_map(|(field, entry)| match entry.field_type() {
            FieldType::Str(options) if options.holds_synonym_rules() => {
                Some((field, entry.name(), options))
            }
            _ => None,
        });
    let Some((field, name, options)) = fields.next() else {
        return Ok(None);
    };
    if let Some((_, other, _)) = fields.next() {
        return Err(TantivyError::SchemaError(format!(
            "a schema can hold synonym rules in one field, found '{name}' and '{other}'"
        )));
    }
    if options.get_indexing_options().is_none() {
        return Err(TantivyError::SchemaError(format!(
            "synonym rule field '{name}' must be indexed, since its tokenizer analyzes the rules"
        )));
    }
    Ok(Some(field))
}

impl SegmentPlugin for SynonymPlugin {
    fn extensions(&self) -> &[&str] {
        &[SYNONYM_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        let rules = match synonym_rule_field(&ctx.segment.schema())? {
            None => None,
            Some(field) => Some(RuleField {
                field,
                analyzer: ctx.segment.index().tokenizer_for_field(field)?,
            }),
        };
        Ok(Box::new(SynonymWriter {
            rules,
            groups: Vec::new(),
            keys: BTreeMap::new(),
            mem_usage: 0,
        }))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        if synonym_rule_field(ctx.schema)?.is_none() {
            return Ok(());
        }
        merge_dictionaries(&ctx)
    }
}

struct RuleField {
    field: Field,
    analyzer: TextAnalyzer,
}

struct Group {
    doc: DocId,
    alternatives: Vec<Vec<String>>,
}

/// Compiles each rule when its document is added, so the writer holds the compiled entries and
/// not the rule text.
struct SynonymWriter {
    /// `None` when the schema holds no rules, and the segment then gets no component.
    rules: Option<RuleField>,
    /// One group per rule, in document order.
    groups: Vec<Group>,
    /// Each key with the indexes of its groups.
    keys: BTreeMap<Vec<u8>, Vec<usize>>,
    mem_usage: usize,
}

impl SynonymWriter {
    fn add_value(&mut self, doc: DocId, value: ErasedValue<'_>) -> crate::Result<()> {
        match value {
            ErasedValue::Leaf(ReferenceValueLeaf::Str(text)) => self.add_text(doc, text),
            ErasedValue::Leaf(ReferenceValueLeaf::PreTokStr(_)) => Err(
                TantivyError::InvalidArgument("a synonym rule can't be pre-tokenized".to_string()),
            ),
            ErasedValue::Array(values) => {
                for value in values {
                    self.add_value(doc, value)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn add_text(&mut self, doc: DocId, text: &str) -> crate::Result<()> {
        let Some(rule_field) = self.rules.as_mut() else {
            return Ok(());
        };
        let invalid =
            |err| TantivyError::InvalidArgument(format!("invalid synonym rule '{text}': {err}"));
        let Some(rule) = SynonymRule::parse(text).map_err(invalid)? else {
            return Ok(());
        };
        if let Some(rule) = rule.compile(&mut rule_field.analyzer).map_err(invalid)? {
            self.add_rule(doc, rule);
        }
        Ok(())
    }

    fn add_rule(&mut self, doc: DocId, rule: CompiledRule) {
        let group = self.groups.len();
        for key in rule.keys {
            let key = encode_key(&key);
            let key_len = key.len();
            self.keys
                .entry(key)
                .or_insert_with(|| {
                    self.mem_usage += key_len + size_of::<(Vec<u8>, Vec<usize>)>();
                    Vec::new()
                })
                .push(group);
            self.mem_usage += size_of::<usize>();
        }
        self.mem_usage += size_of::<Group>()
            + rule
                .alternatives
                .iter()
                .map(|alternative| alternative_mem_usage(alternative))
                .sum::<usize>();
        self.groups.push(Group {
            doc,
            alternatives: rule.alternatives,
        });
    }
}

fn alternative_mem_usage(alternative: &[String]) -> usize {
    size_of::<Vec<String>>()
        + alternative
            .iter()
            .map(|token| size_of::<String>() + token.len())
            .sum::<usize>()
}

impl PluginWriter for SynonymWriter {
    fn add_document(
        &mut self,
        doc_id: DocId,
        doc: &dyn ErasedDocument,
        _schema: &Schema,
    ) -> crate::Result<()> {
        let Some(rule_field) = self.rules.as_ref().map(|rules| rules.field) else {
            return Ok(());
        };
        for (field, value) in doc.erased_fields() {
            if field == rule_field {
                self.add_value(doc_id, value)?;
            }
        }
        Ok(())
    }

    fn serialize(
        self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&DocIdMapping>,
    ) -> crate::Result<()> {
        if self.rules.is_none() {
            return Ok(());
        }
        let SynonymWriter {
            mut groups, keys, ..
        } = *self;
        if let Some(doc_id_map) = doc_id_map {
            for group in &mut groups {
                group.doc = doc_id_map.get_new_doc_id(group.doc);
            }
        }
        let write = segment.open_write(SegmentComponent::Custom(SYNONYM_EXT.to_string()))?;
        let mut file = SynonymFileWriter::new(write);
        let mut refs = Vec::with_capacity(groups.len());
        for group in &groups {
            refs.push(file.add_group(group.doc, &group.alternatives)?);
        }
        let mut writer = file.into_keys();
        let mut key_refs: Vec<GroupRef> = Vec::new();
        for (key, key_groups) in &keys {
            key_refs.clear();
            key_refs.extend(key_groups.iter().map(|&group| refs[group]));
            key_refs.sort_by_key(|group| group.doc);
            writer.insert(key, &key_refs)?;
        }
        writer.finish()?.terminate()?;
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        self.mem_usage
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn check_cancel(ctx: &PluginMergeContext) -> crate::Result<()> {
    if ctx.cancel.wants_cancel() {
        return Err(TantivyError::Cancelled);
    }
    Ok(())
}

/// Merges the dictionaries of the source segments. A group of an alive rule is copied as bytes
/// with its document remapped, so a merge needs neither the rule text nor the tokenizer. A group
/// of a deleted rule is dropped, and so is a key left with no group.
fn merge_dictionaries(ctx: &PluginMergeContext) -> crate::Result<()> {
    let mut target_doc_ids: Vec<Vec<Option<DocId>>> = ctx
        .readers
        .iter()
        .map(|reader| vec![None; reader.max_doc() as usize])
        .collect();
    for (target_doc_id, source) in ctx.doc_id_mapping.iter_source_doc_addrs().enumerate() {
        target_doc_ids[source.segment_ord as usize][source.doc_id as usize] =
            Some(target_doc_id as DocId);
    }
    let target_doc = |segment_ord: usize, doc: DocId| -> crate::Result<Option<DocId>> {
        target_doc_ids[segment_ord]
            .get(doc as usize)
            .copied()
            .ok_or_else(|| {
                DataCorruption::comment_only("a synonym rule names a document outside its segment")
                    .into()
            })
    };

    let mut sources = Vec::new();
    for (segment_ord, reader) in ctx.readers.iter().enumerate() {
        if let Some(dictionary) = reader.synonym_dictionary()? {
            sources.push((segment_ord, dictionary));
        }
    }

    let write = ctx
        .target_segment
        .open_write(SegmentComponent::Custom(SYNONYM_EXT.to_string()))?;
    let mut file = SynonymFileWriter::new(write);
    // For each source, the target group of each alive source group, by source offset.
    let mut moved: Vec<HashMap<u64, GroupRef>> = Vec::with_capacity(sources.len());
    for (segment_ord, dictionary) in &sources {
        let section = dictionary.read_groups()?;
        let mut source_moved = HashMap::new();
        for group in groups(&section) {
            check_cancel(ctx)?;
            let group = group?;
            if let Some(doc) = target_doc(*segment_ord, group.doc)? {
                source_moved.insert(group.offset, file.add_encoded_group(doc, group.payload)?);
            }
        }
        moved.push(source_moved);
    }

    let mut readers = Vec::with_capacity(sources.len());
    for (_, dictionary) in &sources {
        readers.push(dictionary.read_keys()?);
    }
    let mut heap = BinaryHeap::with_capacity(readers.len());
    for (source, reader) in readers.iter_mut().enumerate() {
        if reader.advance()? {
            heap.push(Reverse((reader.key().to_vec(), source)));
        }
    }
    let mut writer = file.into_keys();
    let mut merged: Vec<GroupRef> = Vec::new();
    while let Some(Reverse((key, mut source))) = heap.pop() {
        check_cancel(ctx)?;
        merged.clear();
        loop {
            let segment_ord = sources[source].0;
            let reader = &mut readers[source];
            for group in reader.groups()? {
                let Some(doc) = target_doc(segment_ord, group.doc)? else {
                    continue;
                };
                let target = moved[source]
                    .get(&group.offset())
                    .filter(|target| target.doc == doc)
                    .ok_or_else(|| {
                        DataCorruption::comment_only(
                            "a synonym key names a group that is not in the group section",
                        )
                    })?;
                merged.push(*target);
            }
            if reader.advance()? {
                heap.push(Reverse((reader.key().to_vec(), source)));
            }
            match heap.peek() {
                Some(Reverse((next, _))) if *next == key => {
                    source = heap.pop().unwrap().0 .1;
                }
                _ => break,
            }
        }
        if !merged.is_empty() {
            merged.sort_by_key(|group| group.doc);
            writer.insert(&key, &merged)?;
        }
    }
    writer.finish()?.terminate()?;
    Ok(())
}
