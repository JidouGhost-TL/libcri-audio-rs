//! Standalone pure-Rust CRI Middleware `ADX` and `HCA` audio codec.
//!
//! `libcri-audio` is a clean-room, zero-dependency implementation of the
//! CRI ADX ADPCM codec (versions 3, 4 and 5) and the CRI HCA codec
//! (versions 1.3, 2.0 and 3.0 header dialects), built from the published
//! framing observable in CRI files: the `0x8000` / `"(c)CRI"` ADX header,
//! the chunked `HCA\0` header with its CRC-16, and deterministic
//! fixed-point / IMDCT decoding. No copyrighted audio bytes are included.
//!
//! # Architecture
//! - [`adx`]: zero-allocation [`adx::AdxHeader`] / [`adx::AdxView`] parsers,
//!   a streaming [`adx::AdxDecoder`], an [`adx::AdxEncoder`] for round-trip
//!   synthesis, and type 8 / type 9 LCG encryption key streams.
//! - [`hca`]: chunked [`hca::HcaHeader`] / [`hca::HcaView`] parsers with
//!   CRC-16 validation, keyed cipher tables (types 0, 1 and 56), and a
//!   streaming [`hca::HcaDecoder`] performing bit reading, ATH / intensity /
//!   scalefactor unpacking, spectral dequantization and IMDCT +
//!   overlap-add windowing to `i16` / `f32` planar PCM.
//! - [`crc`]: CRC-16/BUYPASS (poly `0x8005`) used by HCA headers and frames.
//!
//! The hot ADX decode loop performs no heap allocation: per-channel filter
//! history lives in fixed-size arrays inside the decoder. HCA allocates its
//! IMDCT tables once at decoder construction, never per frame.
//!
//! # Example
//! ```no_run
//! use libcri_audio::{AdxDecoder, HcaDecoder};
//! # let adx_bytes: &[u8] = &[];
//! # let hca_bytes: &[u8] = &[];
//! let mut adx = AdxDecoder::new(adx_bytes).unwrap();
//! let mut pcm = [0i16; 64];
//! let _frames = adx.decode_i16(&mut pcm).unwrap();
//! let mut hca = HcaDecoder::new(hca_bytes).unwrap();
//! let _frames = hca.decode_i16(&mut pcm).unwrap();
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod adx;
pub mod crc;
pub mod hca;
pub mod math;

use core::fmt;

/// Result alias for `libcri-audio` operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Error type returned by fallible `libcri-audio` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input ended before the expected header, chunk or frame completed.
    UnexpectedEof,
    /// Magic bytes did not match (`0x8000` for ADX, `HCA\0` for HCA).
    InvalidMagic,
    /// The `"(c)CRI"` copyright terminator was missing from an ADX header.
    MissingCopyright,
    /// A header field held an out-of-range or inconsistent value.
    InvalidField {
        /// Name of the offending field.
        what: &'static str,
        /// Offending value.
        value: u64,
    },
    /// Version byte / field is not supported by this crate.
    UnsupportedVersion(u32),
    /// Encoding, cipher or chunk type is not supported.
    Unsupported(&'static str),
    /// CRC-16 validation failed.
    CrcMismatch {
        /// CRC stored in the stream.
        stored: u16,
        /// CRC computed over the covered bytes.
        computed: u16,
    },
    /// Caller-supplied output buffer is smaller than required.
    BufferTooSmall {
        /// Minimum number of elements required.
        needed: usize,
        /// Actual buffer length provided.
        actual: usize,
    },
    /// Invalid parameter supplied by the caller (key, scale, dimensions).
    InvalidParameter(&'static str),
    /// Frame bitstream desynchronised (ran out of bits or bad values).
    BitstreamDesync,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of CRI audio input"),
            Self::InvalidMagic => write!(f, "invalid CRI audio magic"),
            Self::MissingCopyright => write!(f, "missing (c)CRI copyright terminator"),
            Self::InvalidField { what, value } => {
                write!(f, "invalid field {what}: {value:#x}")
            }
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v:#x}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::CrcMismatch { stored, computed } => {
                write!(
                    f,
                    "CRC mismatch: stored {stored:#06x}, computed {computed:#06x}"
                )
            }
            Self::BufferTooSmall { needed, actual } => {
                write!(f, "buffer too small: needed {needed}, got {actual}")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::BitstreamDesync => write!(f, "bitstream desynchronization"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub use adx::{AdxDecoder, AdxEncoder, AdxEncoding, AdxHeader, AdxLoop, AdxView, EncryptionType};
pub use crc::crc16;
pub use hca::{
    HcaChannelType, HcaCipher, HcaComp, HcaDecoder, HcaEncoder, HcaFrameHeader, HcaHeader, HcaLoop,
    HcaView, ImdctTables,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_and_sync() {
        assert_send_sync::<Error>();
        assert_send_sync::<AdxHeader>();
        assert_send_sync::<HcaHeader>();
        assert_send_sync::<HcaCipher>();
    }
}
