use std::error::Error;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

/// Bytes value parser used by [`QueryParser`].
pub trait BytesParser {
    /// Parses the provided string into an `Vec<u8>`.
    fn parse(&self, s: &str) -> Result<Vec<u8>, Box<dyn Error>>;
}

/// Bytes parses that uses the base64 encoding to read values.
#[derive(Default)]
pub struct Base64Parser {}

impl BytesParser for Base64Parser {
    fn parse(&self, s: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        BASE64.decode(s).map_err(|e| Box::new(e).into())
    }
}
