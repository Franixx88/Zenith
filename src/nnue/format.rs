//! Portable version dispatch: legacy v1/v2 and mirrored layered v3.
use super::*;
use std::io::Read;
use std::path::Path;

pub const FORMAT_VERSION: u32 = 1;
pub const FORMAT_VERSION_V2: u32 = 2;
const MAGIC: &[u8; 8] = b"ZNNUE\0\0\0";
const HEADER: usize = 64;
const PAYLOAD: usize = HIDDEN * 2 + INPUTS * HIDDEN * 2 + 2 * HIDDEN * 2 + 4;
pub const FILE_BYTES: usize = HEADER + PAYLOAD;
const V2_PAYLOAD: usize = PAYLOAD + (KING_BUCKETS - 1) * INPUTS * HIDDEN * 2;
pub const V2_FILE_BYTES: usize = HEADER + V2_PAYLOAD;
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
        file.take(super::v3::FILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("NNUE read: {e}"))?;
        Self::from_bytes(&bytes)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("NNUE write: {e}"))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        if let Some(p) = self.v3_parameters() {
            return p.to_bytes();
        }
        let mut bytes = Vec::with_capacity(self.file_bytes());
        bytes.extend_from_slice(MAGIC);
        let mut fields = FIELDS;
        fields[0] = self.format_version();
        fields[7] = self.king_buckets() as u32;
        for field in fields {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        bytes.extend_from_slice(&((self.file_bytes() - HEADER) as u64).to_le_bytes());
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
        if bytes.len() < HEADER {
            return Err("NNUE truncated header".into());
        }
        if &bytes[..8] != MAGIC {
            return Err("NNUE magic mismatch".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version == 3 {
            return Self::new_v3(super::v3::Parameters::from_bytes(bytes)?);
        }
        let (file_bytes, buckets) = match version {
            FORMAT_VERSION => (FILE_BYTES, 1),
            FORMAT_VERSION_V2 => (V2_FILE_BYTES, KING_BUCKETS),
            _ => return Err("NNUE unsupported format version".into()),
        };
        if bytes.len() != file_bytes {
            return Err(format!(
                "NNUE v{version} requires exactly {file_bytes} bytes"
            ));
        }
        let mut fields = FIELDS;
        fields[0] = version;
        fields[7] = buckets as u32;
        for (index, expected) in fields.iter().enumerate() {
            let offset = 8 + 4 * index;
            if u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) != *expected {
                return Err(format!("NNUE unsupported header field {index}"));
            }
        }
        let read_u64 = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        if read_u64(40) != (file_bytes - HEADER) as u64 || read_u64(56) != 0 {
            return Err("NNUE invalid payload size or reserved header".into());
        }
        if read_u64(48) != checksum(bytes) {
            return Err("NNUE checksum mismatch".into());
        }
        let mut offset = HEADER;
        let mut parameters = Parameters::default();
        parameters
            .feature_weights
            .resize(buckets * INPUTS * HIDDEN, 0);
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
pub(super) fn checksum(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(0xcbf2_9ce4_8422_2325, |hash, (index, &byte)| {
            (hash ^ u64::from(if (48..56).contains(&index) { 0 } else { byte }))
                .wrapping_mul(0x100_0000_01b3)
        })
}
