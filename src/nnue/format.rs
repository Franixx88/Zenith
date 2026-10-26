//! Portable v1 format, documented in docs/nnue-v1.md. No native memory dumps.
use super::*;
use std::io::Read;
use std::path::Path;

pub const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"ZNNUE\0\0\0";
const HEADER: usize = 64;
const PAYLOAD: usize = HIDDEN * 2 + INPUTS * HIDDEN * 2 + 2 * HIDDEN * 2 + 4;
pub const FILE_BYTES: usize = HEADER + PAYLOAD;
const FIELDS: [u32; 8] = [
    FORMAT_VERSION,
    INPUTS as u32,
    HIDDEN as u32,
    1,
    QA as u32,
    QB as u32,
    CP_SCALE as u32,
    1,
];

impl Network {
    /// Bounded read: malformed files cannot request arbitrary model allocations.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("NNUE open: {e}"))?;
        let mut bytes = Vec::with_capacity(FILE_BYTES);
        file.take(FILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("NNUE read: {e}"))?;
        Self::from_bytes(&bytes)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("NNUE write: {e}"))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FILE_BYTES);
        bytes.extend_from_slice(MAGIC);
        for field in FIELDS {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        bytes.extend_from_slice(&(PAYLOAD as u64).to_le_bytes());
        bytes.extend_from_slice(&[0; 16]); // checksum placeholder, reserved
        for values in [
            self.parameters.feature_bias.as_slice(),
            self.parameters.feature_weights.as_slice(),
            self.parameters.output_weights.as_slice(),
        ] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&self.parameters.output_bias.to_le_bytes());
        let checksum = checksum(&bytes);
        bytes[48..56].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != FILE_BYTES {
            return Err(format!("NNUE v1 requires exactly {FILE_BYTES} bytes"));
        }
        if &bytes[..8] != MAGIC {
            return Err("NNUE magic mismatch".into());
        }
        for (index, expected) in FIELDS.iter().enumerate() {
            let offset = 8 + 4 * index;
            if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) != *expected {
                return Err(format!("NNUE unsupported header field {index}"));
            }
        }
        let read_u64 = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        if read_u64(40) != PAYLOAD as u64 || read_u64(56) != 0 {
            return Err("NNUE invalid payload size or reserved header".into());
        }
        if read_u64(48) != checksum(bytes) {
            return Err("NNUE checksum mismatch".into());
        }
        let mut offset = HEADER;
        let mut parameters = Parameters::default();
        for values in [
            parameters.feature_bias.as_mut_slice(),
            parameters.feature_weights.as_mut_slice(),
            parameters.output_weights.as_mut_slice(),
        ] {
            for value in values {
                *value = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
        }
        parameters.output_bias = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        Self::new(parameters)
    }
}

// FNV-1a64 over the entire file with checksum bytes treated as zero. Integrity
// check only, not authentication. Architecture and quantization are covered too.
fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(0xcbf2_9ce4_8422_2325, |hash, (index, &byte)| {
            (hash ^ u64::from(if (48..56).contains(&index) { 0 } else { byte }))
                .wrapping_mul(0x100_0000_01b3)
        })
}
