use std::io::{self, Write};

use crate::Score;
use crate::directory::OwnedBytes;
use crate::postings::skip::{decode_block_wand_max_tf, encode_block_wand_max_tf};
use crate::query::Bm25Weight;

pub(crate) const SUBBLOCK_SIZE: usize = 16;
const MAGIC: [u8; 10] = [127, 127, 127, 127, 127, 127, 127, 127, 127, 129];

#[derive(Clone, Copy, Debug)]
pub(crate) struct SubblockSummary {
    min_norm: u8,
    max_tf: u8,
}

impl Default for SubblockSummary {
    fn default() -> Self {
        Self {
            min_norm: u8::MAX,
            max_tf: 0,
        }
    }
}

impl SubblockSummary {
    pub(crate) fn record(&mut self, norm: u8, tf: u32) {
        self.min_norm = self.min_norm.min(norm);
        self.max_tf = self.max_tf.max(encode_block_wand_max_tf(tf));
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            min_norm: bytes[0],
            max_tf: bytes[1],
        }
    }

    pub(crate) fn bound(self, weight: &Bm25Weight) -> Score {
        weight.max_score_for_min_norm(self.min_norm, decode_block_wand_max_tf(self.max_tf))
    }
}

pub(crate) fn write_summaries(
    summaries: &[SubblockSummary],
    output: &mut impl Write,
) -> io::Result<()> {
    if !summaries.is_empty() {
        output.write_all(&MAGIC)?;
        for summary in summaries {
            output.write_all(&[summary.min_norm, summary.max_tf])?;
        }
    }
    Ok(())
}

pub(crate) fn read_summaries(
    doc_freq: u32,
    mut bytes: OwnedBytes,
) -> io::Result<(OwnedBytes, OwnedBytes)> {
    if !bytes.as_slice().starts_with(&MAGIC) {
        return Ok((OwnedBytes::empty(), bytes));
    }
    bytes.advance(MAGIC.len());
    let len = (doc_freq as usize).div_ceil(SUBBLOCK_SIZE) * 2;
    if len > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated subblock summaries",
        ));
    }
    Ok(bytes.split(len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bm25Params;

    #[test]
    fn conservative_across_scoring_parameters() {
        let values = [(7, 1), (49, 3), (0, 2), (255, 255), (254, u32::MAX)];
        for pairs in [&values[..2], &values[..3], &values[3..]] {
            let mut summary = SubblockSummary::default();
            for &(norm, tf) in pairs {
                summary.record(norm, tf);
            }
            for avg in [0.1, 1.0, 17.0, 1000.0, 1e9] {
                for k1 in [0.0, 0.1, 1.2, 4.0] {
                    for b in [0.0, 0.25, 0.75, 1.0] {
                        for boost in [-1.0, 0.0, 1.0, 3.0] {
                            let weight =
                                Bm25Weight::for_one_term(100, 10000, avg, Bm25Params::new(k1, b))
                                    .boost_by(boost);
                            for &(norm, tf) in pairs {
                                assert!(summary.bound(&weight) >= weight.score(norm, tf));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn framing_legacy_and_truncation() {
        let legacy = OwnedBytes::new(vec![128, 129, 130]);
        let (summaries, data) = read_summaries(1, legacy.clone()).unwrap();
        assert!(summaries.is_empty());
        assert_eq!(data.as_slice(), legacy.as_slice());
        let mut encoded = Vec::new();
        write_summaries(&[SubblockSummary::default(); 2], &mut encoded).unwrap();
        encoded.extend_from_slice(legacy.as_slice());
        let (summaries, data) = read_summaries(17, OwnedBytes::new(encoded)).unwrap();
        assert_eq!(summaries.len(), 4);
        assert_eq!(data.as_slice(), legacy.as_slice());
        assert!(read_summaries(17, OwnedBytes::new(MAGIC.to_vec())).is_err());
    }
}
