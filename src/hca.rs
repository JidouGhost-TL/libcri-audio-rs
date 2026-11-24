//! CRI HCA (v1.3 / v2.0 / v3.0) header parsing, cipher, fast FFT IMDCT and frame decoding.
//!
//! # Container & Frame Framing (Real CRI HCA Specification)
//! - Header starts with `HCA\0` (tags may have bit 7 set and are masked with `0x7F`),
//!   big-endian `u16` version (`0x0103`, `0x0200`, `0x0300`) and `u16` header size,
//!   followed by chunks (`fmt`, `comp`, `dec`, `vbr`, `ath`, `loop`, `ciph`, `rva`,
//!   `comm`, `pad`) and a trailing CRC-16/BUYPASS (`0x8005`) over the entire header.
//! - Each audio frame carries **8 subframes × 128 samples = 1024 samples per channel**,
//!   framed with a 16-bit sync word (`0xFFFF`), 9-bit acceptable noise level `a`,
//!   7-bit evaluation boundary `b`, per-channel 3-bit `delta_bits` + 6-bit delta-coded
//!   scalefactors (`0..=63`), 4-bit stereo intensity values, optional 6-bit HFR
//!   scalefactors, 8 subframes of quantized MDCT spectra, and a trailing CRC-16/BUYPASS.
//! - Synthesis uses a **fast 64-point radix-2 FFT-based 128-point IMDCT** (O(N log N),
//!   zero per-frame heap allocation) with a 256-sample Princen-Bradley sine window
//!   and 128-sample overlap-add.
//!
//! # Clean-Room Scope Note
//! All container parsing, CRC-16/BUYPASS verification, frame sync (`0xFFFF`), `a`/`b`,
//! delta-coded 6-bit scalefactors, intensity/HFR header fields, and the O(N log N)
//! FFT-based 128-point IMDCT follow the published CRI HCA framing and textbook DSP.
//! Proprietary CRI static lookup tables (CRI's 653-entry ATH table, scale-to-resolution
//! curve, 7 spectral Huffman codebooks, dequantizer/intensity/HFR tables, and internal
//! Type 56 shuffle constants) are not present in this clean-room crate; clean-room
//! analytical equivalents are used instead and documented on each routine.

use crate::crc::crc16;
use crate::math;
use crate::{Error, Result};

/// Spectral bins per HCA subframe per channel.
pub const SPECTRAL_BINS: usize = 128;
/// Subframes per HCA frame.
pub const SUBFRAMES_PER_FRAME: usize = 8;
/// PCM samples produced per subframe per channel (IMDCT hop).
pub const SAMPLES_PER_SUBFRAME: usize = 128;
/// PCM samples produced per HCA frame per channel (`8 * 128 = 1024`).
pub const SAMPLES_PER_FRAME: usize = SUBFRAMES_PER_FRAME * SAMPLES_PER_SUBFRAME;
/// 16-bit sync word at the start of every decrypted HCA frame.
pub const FRAME_SYNC: u16 = 0xFFFF;
/// Maximum channels supported without per-frame allocation.
pub const MAX_CHANNELS: usize = 8;
/// Maximum accepted frame size in bytes.
pub const MAX_FRAME_SIZE: usize = 4096;

/// Channel coding role within an HCA frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HcaChannelType {
    /// Independently coded mono/discrete channel.
    Discrete,
    /// Primary (left/mid) channel of an intensity-stereo pair.
    StereoPrimary,
    /// Secondary (right/side) channel of an intensity-stereo pair.
    StereoSecondary,
}

/// Codec parameters from the `comp` or `dec` chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcaComp {
    /// Minimum spectral bit resolution.
    pub min_resolution: u8,
    /// Maximum spectral bit resolution.
    pub max_resolution: u8,
    /// Track count.
    pub track_count: u8,
    /// Channel configuration.
    pub channel_config: u8,
    /// Total number of scalefactor bands (up to 128).
    pub total_bands: u8,
    /// Base (independently coded) bands.
    pub base_bands: u8,
    /// Stereo (intensity) bands.
    pub stereo_bands: u8,
    /// Bands per high-frequency reconstruction group.
    pub bands_per_hfr_group: u8,
}

/// Loop points, in frames, plus trim samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcaLoop {
    /// First looped frame.
    pub start_frame: u32,
    /// Frame one past the loop end.
    pub end_frame: u32,
    /// Samples trimmed before the loop start.
    pub pre_samples: u16,
    /// Samples trimmed after the loop end.
    pub post_samples: u16,
}

/// Parsed HCA header (owned).
#[derive(Debug, Clone, PartialEq)]
pub struct HcaHeader {
    /// True when header tags carry high bits set (masked form).
    pub masked: bool,
    /// Raw version (`0x0103`, `0x0200`, `0x0300`, ...).
    pub version: u16,
    /// Header size in bytes (frame data starts here).
    pub header_size: u16,
    /// Chunk names in order, trailing NULs stripped.
    #[cfg(feature = "alloc")]
    pub chunks: alloc::vec::Vec<alloc::string::String>,
    /// Chunk names (fixed storage when `alloc` is disabled).
    #[cfg(not(feature = "alloc"))]
    pub chunks: ([[u8; 4]; 12], usize),
    /// Channel count (`1..=8` for decoding; header allows up to `16`).
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Frame count.
    pub frame_count: u32,
    /// Encoder delay in samples.
    pub encoder_delay: u16,
    /// Encoder padding in samples.
    pub encoder_padding: u16,
    /// Bytes per frame.
    pub frame_size: u16,
    /// Codec parameters, if a `comp`/`dec` chunk was present.
    pub comp: Option<HcaComp>,
    /// ATH curve type, if an `ath` chunk was present.
    pub ath_type: Option<u16>,
    /// VBR maximum frame size field, if a `vbr` chunk was present.
    pub vbr: Option<u16>,
    /// Loop points, if present.
    pub looping: Option<HcaLoop>,
    /// Cipher type: `0` none, `1` keyless, `56` keyed.
    pub cipher_type: u16,
    /// RVA volume, if present.
    pub rva_volume: Option<f32>,
    /// Comment length, if a `comm` chunk was present.
    pub comment_len: Option<usize>,
}

fn unmask4(b: &[u8]) -> [u8; 4] {
    [b[0] & 0x7F, b[1] & 0x7F, b[2] & 0x7F, b[3] & 0x7F]
}
fn be_u16(b: &[u8], at: usize) -> Result<u16> {
    let s = b.get(at..at + 2).ok_or(Error::UnexpectedEof)?;
    Ok(u16::from_be_bytes([s[0], s[1]]))
}
fn be_u32(b: &[u8], at: usize) -> Result<u32> {
    let s = b.get(at..at + 4).ok_or(Error::UnexpectedEof)?;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

const CHUNK_SIZES: [([u8; 4], usize); 10] = [
    (*b"fmt\0", 16),
    (*b"comp", 16),
    (*b"dec\0", 12),
    (*b"vbr\0", 8),
    (*b"ath\0", 6),
    (*b"loop", 16),
    (*b"ciph", 6),
    (*b"rva\0", 8),
    (*b"comm", 5),
    (*b"pad\0", 4),
];

impl HcaHeader {
    /// True if `bytes` starts with the (possibly masked) `HCA\0` magic.
    pub fn is_magic(bytes: &[u8]) -> bool {
        bytes.len() >= 4 && unmask4(&bytes[..4]) == *b"HCA\0"
    }

    /// Parse an HCA header from the start of `bytes` and validate its CRC-16/BUYPASS.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::UnexpectedEof);
        }
        if !Self::is_magic(bytes) {
            return Err(Error::InvalidMagic);
        }
        let masked = bytes[..4] != *b"HCA\0";
        let version = be_u16(bytes, 4)?;
        if !(0x0100..=0x0300).contains(&version) {
            return Err(Error::UnsupportedVersion(version as u32));
        }
        let header_size = be_u16(bytes, 6)?;
        if (header_size as usize) < 10 || header_size as usize > bytes.len() {
            return Err(Error::InvalidField {
                what: "header_size",
                value: header_size as u64,
            });
        }
        let header = &bytes[..header_size as usize];
        if crc16(header) != 0 {
            let stored = be_u16(header, header.len() - 2)?;
            return Err(Error::CrcMismatch {
                stored,
                computed: crc16(&header[..header.len() - 2]),
            });
        }
        let mut h = Self {
            masked,
            version,
            header_size,
            #[cfg(feature = "alloc")]
            chunks: alloc::vec::Vec::new(),
            #[cfg(not(feature = "alloc"))]
            chunks: ([[0u8; 4]; 12], 0),
            channels: 0,
            sample_rate: 0,
            frame_count: 0,
            encoder_delay: 0,
            encoder_padding: 0,
            frame_size: 0,
            comp: None,
            ath_type: None,
            vbr: None,
            looping: None,
            cipher_type: 0,
            rva_volume: None,
            comment_len: None,
        };
        let limit = header_size as usize - 2;
        let mut p = 8usize;
        while p + 4 <= limit {
            let tag = unmask4(&header[p..p + 4]);
            if tag == *b"pad\0" || tag == [0, 0, 0, 0] {
                break;
            }
            let Some(&(_, size)) = CHUNK_SIZES.iter().find(|(t, _)| *t == tag) else {
                return Err(Error::Unsupported("HCA chunk tag"));
            };
            #[cfg(feature = "alloc")]
            h.chunks.push(
                core::str::from_utf8(&tag)
                    .unwrap_or("")
                    .trim_end_matches('\0')
                    .into(),
            );
            #[cfg(not(feature = "alloc"))]
            {
                if h.chunks.1 < 12 {
                    h.chunks.0[h.chunks.1] = tag;
                    h.chunks.1 += 1;
                }
            }
            let body = header.get(p..p + size).ok_or(Error::UnexpectedEof)?;
            match &tag {
                b"fmt\0" => {
                    h.channels = body[4];
                    h.sample_rate =
                        ((body[5] as u32) << 16) | ((body[6] as u32) << 8) | body[7] as u32;
                    h.frame_count = be_u32(body, 8)?;
                    h.encoder_delay = be_u16(body, 12)?;
                    h.encoder_padding = be_u16(body, 14)?;
                }
                b"comp" => {
                    h.frame_size = be_u16(body, 4)?;
                    h.comp = Some(HcaComp {
                        min_resolution: body[6],
                        max_resolution: body[7],
                        track_count: body[8],
                        channel_config: body[9],
                        total_bands: body[10],
                        base_bands: body[11],
                        stereo_bands: body[12],
                        bands_per_hfr_group: body[13],
                    });
                }
                b"dec\0" => {
                    h.frame_size = be_u16(body, 4)?;
                    let total = body[8].saturating_add(1);
                    let stereo_type = body[10];
                    let base_bands = if stereo_type != 0 {
                        body[9].saturating_add(1)
                    } else {
                        total
                    };
                    h.comp = Some(HcaComp {
                        min_resolution: body[6],
                        max_resolution: body[7],
                        track_count: (body[11] >> 4).max(1),
                        channel_config: stereo_type,
                        total_bands: total,
                        base_bands,
                        stereo_bands: total.saturating_sub(base_bands),
                        bands_per_hfr_group: body[11] & 0x0F,
                    });
                }
                b"vbr\0" => h.vbr = Some(be_u16(body, 4)?),
                b"ath\0" => h.ath_type = Some(be_u16(body, 4)?),
                b"loop" => {
                    h.looping = Some(HcaLoop {
                        start_frame: be_u32(body, 4)?,
                        end_frame: be_u32(body, 8)?,
                        pre_samples: be_u16(body, 12)?,
                        post_samples: be_u16(body, 14)?,
                    });
                }
                b"ciph" => h.cipher_type = be_u16(body, 4)?,
                b"rva\0" => {
                    h.rva_volume = Some(f32::from_bits(be_u32(body, 4)?));
                }
                b"comm" => h.comment_len = Some(body[4] as usize),
                _ => unreachable!(),
            }
            p += if tag == *b"comm" {
                5 + body[4] as usize
            } else {
                size
            };
        }
        if h.channels == 0 || h.channels > 16 {
            return Err(Error::InvalidField {
                what: "channels",
                value: h.channels as u64,
            });
        }
        if h.sample_rate == 0 || h.sample_rate > 0x7F_FFFF {
            return Err(Error::InvalidField {
                what: "sample_rate",
                value: h.sample_rate as u64,
            });
        }
        if (h.frame_size as usize) < 8 || h.frame_size as usize > MAX_FRAME_SIZE {
            return Err(Error::InvalidField {
                what: "frame_size",
                value: h.frame_size as u64,
            });
        }
        if !matches!(h.cipher_type, 0 | 1 | 56) {
            return Err(Error::InvalidField {
                what: "cipher_type",
                value: h.cipher_type as u64,
            });
        }
        if let Some(c) = h.comp {
            if c.max_resolution > 31 || c.max_resolution < c.min_resolution {
                return Err(Error::InvalidField {
                    what: "resolution",
                    value: c.max_resolution as u64,
                });
            }
            if c.total_bands as usize > SPECTRAL_BINS
                || (c.base_bands as usize + c.stereo_bands as usize) > SPECTRAL_BINS
            {
                return Err(Error::InvalidField {
                    what: "total_bands",
                    value: c.total_bands as u64,
                });
            }
        }
        Ok(h)
    }

    /// Total samples per channel declared (`frame_count * 1024`).
    pub fn total_samples(&self) -> u64 {
        self.frame_count as u64 * SAMPLES_PER_FRAME as u64
    }

    /// Byte length of the frame stream.
    pub fn data_len(&self) -> usize {
        self.frame_count as usize * self.frame_size as usize
    }

    /// Effective ATH curve type (defaults to `1` for v1.x and `0` for v2.0+ when absent).
    pub fn effective_ath_type(&self) -> u16 {
        self.ath_type
            .unwrap_or(if self.version < 0x0200 { 1 } else { 0 })
    }

    /// Effective total band count (defaults to `128` without a `comp` chunk).
    pub fn bands(&self) -> usize {
        match self.comp {
            Some(c) if c.total_bands > 0 => (c.total_bands as usize).min(SPECTRAL_BINS),
            _ => SPECTRAL_BINS,
        }
    }

    /// Effective `(base_bands, stereo_bands)` counts.
    pub fn band_layout(&self) -> (usize, usize) {
        match self.comp {
            Some(c) if c.total_bands > 0 => {
                let base = (c.base_bands as usize).min(SPECTRAL_BINS);
                let stereo = (c.stereo_bands as usize).min(SPECTRAL_BINS - base);
                (base, stereo)
            }
            _ => (SPECTRAL_BINS, 0),
        }
    }

    /// Number of HFR (high-frequency reconstruction) groups per non-secondary channel.
    pub fn hfr_group_count(&self) -> usize {
        let Some(c) = self.comp else {
            return 0;
        };
        if c.bands_per_hfr_group == 0 {
            return 0;
        }
        let coded = (c.base_bands as usize).saturating_add(c.stereo_bands as usize);
        let hfr_bands = (c.total_bands as usize).saturating_sub(coded);
        let g = c.bands_per_hfr_group as usize;
        (hfr_bands + g - 1) / g
    }

    /// Resolution range `(min_resolution, max_resolution)` (defaults to `1..=15`).
    pub fn resolution_range(&self) -> (u8, u8) {
        match self.comp {
            Some(c) => (c.min_resolution, c.max_resolution),
            None => (1, 15),
        }
    }

    /// Per-channel stereo/discrete coding role.
    pub fn channel_types(&self) -> [HcaChannelType; MAX_CHANNELS] {
        let mut types = [HcaChannelType::Discrete; MAX_CHANNELS];
        let Some(c) = self.comp else {
            return types;
        };
        if c.stereo_bands == 0 {
            return types;
        }
        let ch = (self.channels as usize).min(MAX_CHANNELS);
        let tracks = (c.track_count as usize).max(1);
        let cpt = ch / tracks;
        for t in 0..tracks {
            let base = t * cpt;
            match cpt {
                2 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                }
                3 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                }
                4 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                    if c.channel_config != 0 {
                        types[base + 2] = HcaChannelType::StereoPrimary;
                        types[base + 3] = HcaChannelType::StereoSecondary;
                    }
                }
                5 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                    types[base + 3] = HcaChannelType::StereoPrimary;
                    types[base + 4] = HcaChannelType::StereoSecondary;
                }
                6 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                    types[base + 4] = HcaChannelType::StereoPrimary;
                    types[base + 5] = HcaChannelType::StereoSecondary;
                }
                7 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                    types[base + 4] = HcaChannelType::StereoPrimary;
                    types[base + 5] = HcaChannelType::StereoSecondary;
                }
                8 => {
                    types[base] = HcaChannelType::StereoPrimary;
                    types[base + 1] = HcaChannelType::StereoSecondary;
                    types[base + 4] = HcaChannelType::StereoPrimary;
                    types[base + 5] = HcaChannelType::StereoSecondary;
                    types[base + 6] = HcaChannelType::StereoPrimary;
                    types[base + 7] = HcaChannelType::StereoSecondary;
                }
                _ => {}
            }
        }
        types
    }
}

/// Zero-allocation borrowed view of a complete HCA stream.
#[derive(Debug, Clone, PartialEq)]
pub struct HcaView<'a> {
    /// Parsed header.
    pub header: HcaHeader,
    /// Frame stream bytes.
    pub frames: &'a [u8],
    raw: &'a [u8],
}

impl<'a> HcaView<'a> {
    /// Parse and bound a complete HCA stream.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = HcaHeader::parse(bytes)?;
        let start = header.header_size as usize;
        let end = start
            .checked_add(header.data_len())
            .ok_or(Error::InvalidField {
                what: "data_len",
                value: 0,
            })?;
        let frames = bytes.get(start..end).ok_or(Error::UnexpectedEof)?;
        Ok(Self {
            header,
            frames,
            raw: bytes,
        })
    }

    /// Original bytes.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.raw
    }

    /// One frame by index.
    pub fn frame(&self, i: usize) -> Option<&'a [u8]> {
        let n = self.header.frame_size as usize;
        self.frames.get(i * n..i * n + n)
    }
}

/// HCA byte-substitution cipher (types 0, 1 and 56).
///
/// Both Type 1 and Type 56 preserve anchor bytes `0x00 -> 0x00` and `0xFF -> 0xFF`
/// so the `0xFFFF` frame sync word remains invariant. Type 56 supports CRI's
/// 16-bit AWB container subkey derivation formula:
/// `key' = key * ((subkey << 16) | (!subkey + 2))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcaCipher {
    table: [u8; 256],
    inverse: [u8; 256],
    /// Cipher type this table was built for.
    pub cipher_type: u16,
}

/// Derive the effective 64-bit Type 56 key from a base key and 16-bit AWB subkey.
pub fn derive_type56_subkey(key: u64, subkey: u16) -> u64 {
    if subkey == 0 {
        key
    } else {
        let factor = ((subkey as u64) << 16) | ((!subkey).wrapping_add(2) as u64);
        key.wrapping_mul(factor)
    }
}

impl HcaCipher {
    /// Identity cipher (type 0).
    pub fn none() -> Self {
        let mut table = [0u8; 256];
        for (i, t) in table.iter_mut().enumerate() {
            *t = i as u8;
        }
        Self {
            table,
            inverse: table,
            cipher_type: 0,
        }
    }

    /// Keyless cipher (type 1): fixed affine permutation `(t * 13 + 11) & 0xFF`
    /// with fixed `0x00` and `0xFF` anchors.
    pub fn type1() -> Self {
        let mut table = [0u8; 256];
        table[0] = 0;
        table[255] = 255;
        let mut t = 0u8;
        for slot in table.iter_mut().take(255).skip(1) {
            loop {
                t = t.wrapping_mul(13).wrapping_add(11);
                if t != 0 && t != 255 {
                    *slot = t;
                    break;
                }
            }
        }
        Self::from_forward_table(1, table)
    }

    /// Keyed cipher (type 56) from a 56/64-bit key (no subkey).
    ///
    /// Clean-room bijective permutation over `1..=254` preserving `0x00` and `0xFF`
    /// anchors, driven by the 7 bytes of `key.wrapping_sub(1)`.
    pub fn type56(key: u64) -> Self {
        Self::type56_with_awb_subkey(key, 0)
    }

    /// Keyed cipher (type 56) from a 64-bit key and an explicit 16-bit AWB subkey.
    pub fn type56_with_awb_subkey(key: u64, subkey: u16) -> Self {
        if key == 0 {
            return Self::none();
        }
        let eff = derive_type56_subkey(key, subkey);
        Self::build_type56(eff)
    }

    /// Keyed cipher (type 56) from a 64-bit key and an explicit 64-bit subkey.
    pub fn type56_with_subkey(key: u64, subkey: u64) -> Self {
        let eff = if subkey <= u16::MAX as u64 {
            derive_type56_subkey(key, subkey as u16)
        } else {
            key ^ subkey
        };
        if eff == 0 {
            return Self::none();
        }
        Self::build_type56(eff)
    }

    fn build_type56(key: u64) -> Self {
        let k = key.wrapping_sub(1);
        let kc = k.to_le_bytes();
        // Build two 16-element nibble permutations from the 7 key bytes,
        // then compose them into a full bijection over 1..=254 (0x00/0xFF anchored).
        let mut t1 = [0u8; 16];
        let mut t2 = [0u8; 16];
        for i in 0..16u8 {
            t1[i as usize] = i;
            t2[i as usize] = i;
        }
        let mut s1 = (kc[0] as u32)
            | ((kc[1] as u32) << 8)
            | ((kc[2] as u32) << 16)
            | ((kc[3] as u32) << 24);
        let mut s2 = (kc[4] as u32) | ((kc[5] as u32) << 8) | ((kc[6] as u32) << 16) | 0xA500_0000;
        for i in (1..16usize).rev() {
            s1 = s1.wrapping_mul(0x0019_660D).wrapping_add(0x3C6E_F35F);
            t1.swap(i, ((s1 >> 16) as usize) % (i + 1));
            s2 = s2.wrapping_mul(0x0019_660D).wrapping_add(0x3C6E_F35F);
            t2.swap(i, ((s2 >> 16) as usize) % (i + 1));
        }
        let mut table = [0u8; 256];
        table[0] = 0;
        table[255] = 255;
        let mut out_idx = 1usize;
        for &hi in &t1 {
            for &lo in &t2 {
                let v = (hi << 4) | lo;
                if v != 0 && v != 255 {
                    table[out_idx] = v;
                    out_idx += 1;
                }
            }
        }
        Self::from_forward_table(56, table)
    }

    fn from_forward_table(cipher_type: u16, table: [u8; 256]) -> Self {
        let mut inverse = [0u8; 256];
        for (i, &t) in table.iter().enumerate() {
            inverse[t as usize] = i as u8;
        }
        Self {
            table,
            inverse,
            cipher_type,
        }
    }

    /// Build the cipher declared by a header (`key` used only for type 56).
    pub fn for_header(header: &HcaHeader, key: u64) -> Self {
        Self::for_header_with_subkey(header, key, 0)
    }

    /// Build the cipher declared by a header with an explicit 16-bit AWB subkey.
    pub fn for_header_with_subkey(header: &HcaHeader, key: u64, subkey: u16) -> Self {
        match header.cipher_type {
            1 => Self::type1(),
            56 => Self::type56_with_awb_subkey(key, subkey),
            _ => Self::none(),
        }
    }

    /// Encrypt (forward substitution) in place.
    pub fn apply(&self, data: &mut [u8]) {
        if self.cipher_type == 0 {
            return;
        }
        for b in data.iter_mut() {
            *b = self.table[*b as usize];
        }
    }

    /// Decrypt (inverse substitution) in place.
    pub fn apply_inverse(&self, data: &mut [u8]) {
        if self.cipher_type == 0 {
            return;
        }
        for b in data.iter_mut() {
            *b = self.inverse[*b as usize];
        }
    }

    /// Forward table entry.
    pub fn table_entry(&self, b: u8) -> u8 {
        self.table[b as usize]
    }
}

/// Build the 128-bin ATH (Absolute Threshold of Hearing) curve table.
///
/// For `ath_type == 0`, returns all zeros. For `ath_type == 1`, computes a
/// clean-room analytical psychoacoustic threshold curve scaled by `sample_rate`
/// (low-frequency rolloff below ~400 Hz and high-frequency rolloff above ~10 kHz).
pub fn build_ath_table(ath_type: u16, sample_rate: u32) -> [u8; SPECTRAL_BINS] {
    let mut table = [0u8; SPECTRAL_BINS];
    if ath_type == 0 || sample_rate == 0 {
        return table;
    }
    let nyquist = sample_rate as f32 * 0.5;
    for (bin, slot) in table.iter_mut().enumerate() {
        let freq_hz = ((bin as f32 + 0.5) / SPECTRAL_BINS as f32) * nyquist;
        let f_khz = (freq_hz / 1000.0).max(0.02);
        // Clean-room psychoacoustic U-shaped threshold curve in [0, 120]:
        // lowest threshold around 2..5 kHz, rising at sub-bass and >12 kHz.
        let low_rise = if f_khz < 0.5 {
            (0.5 / f_khz).min(8.0) * 6.0
        } else {
            0.0
        };
        let high_rise = if f_khz > 8.0 {
            let d = (f_khz - 8.0) / 8.0;
            d * d * 28.0
        } else {
            0.0
        };
        let v = (low_rise + high_rise).clamp(0.0, 120.0);
        *slot = v as u8;
    }
    table
}

/// ATH attenuation (in resolution bits) for a spectral bin (convenience helper).
pub fn ath_attenuation(ath_type: u16, bin: usize) -> u8 {
    match ath_type {
        0 => 0,
        1 => (bin / 32).min(7) as u8,
        _ => (7usize.saturating_sub(bin / 18)).min(7) as u8,
    }
}

/// MSB-first bit reader over a frame.
#[derive(Debug, Clone, Copy)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Current bit offset from the start of the buffer.
    pub fn position(&self) -> usize {
        self.bit_pos
    }

    /// Bits remaining.
    pub fn remaining(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.bit_pos)
    }

    /// Read `n` (`<= 32`) bits as an unsigned value.
    pub fn read(&mut self, n: u8) -> Result<u32> {
        if n > 32 {
            return Err(Error::InvalidParameter("bit width"));
        }
        if n == 0 {
            return Ok(0);
        }
        if self.remaining() < n as usize {
            return Err(Error::BitstreamDesync);
        }
        let mut v: u32 = 0;
        let mut left = n as usize;
        while left > 0 {
            let byte_idx = self.bit_pos >> 3;
            let bit_off = self.bit_pos & 7;
            let take = left.min(8 - bit_off);
            let shift = 8 - bit_off - take;
            let mask = ((1u16 << take) - 1) as u8;
            let chunk = (self.data[byte_idx] >> shift) & mask;
            v = (v << take) | chunk as u32;
            self.bit_pos += take;
            left -= take;
        }
        Ok(v)
    }

    /// Read `n` bits as a sign-extended signed value (`n == 0` yields `0`).
    pub fn read_signed(&mut self, n: u8) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.read(n)? as i32;
        let sign = 1i32 << (n - 1);
        Ok(if v & sign != 0 { v - (1i32 << n) } else { v })
    }
}

/// MSB-first bit writer (fixed 4096-byte scratch, no allocation).
#[derive(Debug, Clone)]
pub struct BitWriter {
    data: [u8; MAX_FRAME_SIZE],
    bit_pos: usize,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    /// Create an empty bit writer.
    pub fn new() -> Self {
        Self {
            data: [0u8; MAX_FRAME_SIZE],
            bit_pos: 0,
        }
    }

    /// Write the low `n` bits of `v`.
    pub fn write(&mut self, v: u32, n: u8) -> Result<()> {
        for i in 0..n {
            let idx = self.bit_pos / 8;
            if idx >= self.data.len() {
                return Err(Error::BufferTooSmall {
                    needed: idx + 1,
                    actual: self.data.len(),
                });
            }
            let bit = ((v >> (n - 1 - i)) & 1) as u8;
            if bit != 0 {
                self.data[idx] |= 0x80 >> (self.bit_pos % 8);
            }
            self.bit_pos += 1;
        }
        Ok(())
    }

    /// Write a signed value in `n` two's-complement bits.
    pub fn write_signed(&mut self, v: i32, n: u8) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let masked = (v as u32) & ((1u32 << n) - 1);
        self.write(masked, n)
    }

    /// Bytes used so far (rounded up).
    pub fn byte_len(&self) -> usize {
        (self.bit_pos + 7) / 8
    }

    /// Written bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.byte_len().min(self.data.len())]
    }
}

/// Parsed CRI HCA frame header (sync `0xFFFF`, `a`, `b`, delta-coded 6-bit
/// scalefactors, 4-bit intensity values, and 6-bit HFR scalefactors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcaFrameHeader {
    /// Frame sync word (must be `0xFFFF`).
    pub sync: u16,
    /// 9-bit acceptable noise level `a` (`0..=511`).
    pub acceptable_noise_level: u16,
    /// 7-bit evaluation boundary `b` (`0..=127`).
    pub evaluation_boundary: u8,
    /// Per-channel 6-bit scalefactors (`0..=63`).
    pub scale_factors: [[u8; SPECTRAL_BINS]; MAX_CHANNELS],
    /// Per-channel 4-bit intensity stereo indices (8 per frame).
    pub intensity: [[u8; SUBFRAMES_PER_FRAME]; MAX_CHANNELS],
    /// Per-channel 6-bit HFR group scalefactors.
    pub hfr_scale_factors: [[u8; 64]; MAX_CHANNELS],
    /// Bit position where subframe 0 spectral data starts.
    pub header_bits: usize,
}

impl HcaFrameHeader {
    /// Parse a decrypted HCA frame's header (without the trailing 2-byte CRC).
    pub fn parse(payload: &[u8], header: &HcaHeader) -> Result<Self> {
        let mut br = BitReader::new(payload);
        let sync = br.read(16)? as u16;
        if sync != FRAME_SYNC {
            return Err(Error::BitstreamDesync);
        }
        let acceptable_noise_level = br.read(9)? as u16;
        let evaluation_boundary = br.read(7)? as u8;
        let ch = (header.channels as usize).min(MAX_CHANNELS);
        let (base_bands, stereo_bands) = header.band_layout();
        let coded_primary = (base_bands + stereo_bands).min(SPECTRAL_BINS);
        let ch_types = header.channel_types();
        let hfr_groups = header.hfr_group_count().min(64);

        let mut scale_factors = [[0u8; SPECTRAL_BINS]; MAX_CHANNELS];
        let mut intensity = [[8u8; SUBFRAMES_PER_FRAME]; MAX_CHANNELS];
        let mut hfr_scale_factors = [[0u8; 64]; MAX_CHANNELS];

        for c in 0..ch {
            let count = match ch_types[c] {
                HcaChannelType::StereoSecondary => base_bands,
                _ => coded_primary,
            };
            let delta_bits = br.read(3)? as u8;
            if delta_bits >= 6 {
                for slot in scale_factors[c].iter_mut().take(count) {
                    *slot = br.read(6)? as u8;
                }
            } else if delta_bits > 0 && count > 0 {
                scale_factors[c][0] = br.read(6)? as u8;
                let escape = (1u32 << delta_bits) - 1;
                let bias = (1i32 << (delta_bits - 1)) - 1;
                for b in 1..count {
                    let delta = br.read(delta_bits)?;
                    if delta == escape {
                        scale_factors[c][b] = br.read(6)? as u8;
                    } else {
                        let prev = scale_factors[c][b - 1] as i32;
                        let val = prev + delta as i32 - bias;
                        if !(0..=63).contains(&val) {
                            return Err(Error::BitstreamDesync);
                        }
                        scale_factors[c][b] = val as u8;
                    }
                }
            }
            if ch_types[c] == HcaChannelType::StereoSecondary {
                if header.version >= 0x0200 {
                    for slot in &mut intensity[c] {
                        *slot = br.read(4)? as u8;
                    }
                } else {
                    let v = br.read(4)? as u8;
                    intensity[c] = [v; SUBFRAMES_PER_FRAME];
                }
            }
            if hfr_groups > 0 && ch_types[c] != HcaChannelType::StereoSecondary {
                for slot in hfr_scale_factors[c].iter_mut().take(hfr_groups) {
                    *slot = br.read(6)? as u8;
                }
            }
        }

        Ok(Self {
            sync,
            acceptable_noise_level,
            evaluation_boundary,
            scale_factors,
            intensity,
            hfr_scale_factors,
            header_bits: br.position(),
        })
    }
}

fn clamp16(v: f32) -> i16 {
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        -32768
    } else {
        v as i16
    }
}

/// Princen-Bradley sine window of length 256 used by the 128-point IMDCT.
pub fn sine_window() -> [f32; 256] {
    let mut w = [0f32; 256];
    for (n, x) in w.iter_mut().enumerate() {
        *x = math::sin_f32(math::PI * (n as f32 + 0.5) / 256.0);
    }
    w
}

/// Precomputed tables for the O(N log N) 64-point FFT-based 128-point IMDCT.
///
/// Zero heap allocation; constructed once per decoder/encoder.
#[derive(Debug, Clone)]
pub struct ImdctTables {
    /// 256-sample Princen-Bradley sine window.
    pub window: [f32; 256],
    /// Pre/post-twiddle cosine `cos(2*pi*(k + 1/8) / 256)` for `k = 0..64`.
    twiddle_cos: [f32; 64],
    /// Pre/post-twiddle sine `sin(2*pi*(k + 1/8) / 256)` for `k = 0..64`.
    twiddle_sin: [f32; 64],
    /// 64-point radix-2 FFT twiddle cosines (`cos(2*pi*j/64)` for `j = 0..32`).
    fft_cos: [f32; 32],
    /// 64-point radix-2 FFT twiddle sines (`sin(2*pi*j/64)` for `j = 0..32`).
    fft_sin: [f32; 32],
    /// 6-bit bit-reversal permutation for 64 points.
    bitrev: [u8; 64],
}

impl Default for ImdctTables {
    fn default() -> Self {
        Self::new()
    }
}

impl ImdctTables {
    /// Precompute all twiddle factors, bit-reversal indices, and the sine window.
    pub fn new() -> Self {
        let window = sine_window();
        let mut twiddle_cos = [0.0f32; 64];
        let mut twiddle_sin = [0.0f32; 64];
        for k in 0..64usize {
            let ang = (2.0 * math::PI / 256.0) * (k as f32 + 0.125);
            twiddle_cos[k] = math::cos_f32(ang);
            twiddle_sin[k] = math::sin_f32(ang);
        }
        let mut fft_cos = [0.0f32; 32];
        let mut fft_sin = [0.0f32; 32];
        for j in 0..32usize {
            let ang = (2.0 * math::PI / 64.0) * (j as f32);
            fft_cos[j] = math::cos_f32(ang);
            fft_sin[j] = math::sin_f32(ang);
        }
        let mut bitrev = [0u8; 64];
        for (i, slot) in bitrev.iter_mut().enumerate() {
            let mut x = i;
            let mut r = 0usize;
            for _ in 0..6 {
                r = (r << 1) | (x & 1);
                x >>= 1;
            }
            *slot = r as u8;
        }
        Self {
            window,
            twiddle_cos,
            twiddle_sin,
            fft_cos,
            fft_sin,
            bitrev,
        }
    }

    /// Fast O(N log N) 128-point IMDCT -> 256 windowed time-domain samples.
    ///
    /// Computes `out[n] = window[n] * (1/64) * sum_{k=0..127} X[k] * cos((pi/128)*(n + 64.5)*(k + 0.5))`
    /// using a 64-point radix-2 complex FFT (192 butterflies + 128 twiddles).
    pub fn imdct(&self, input: &[f32; SPECTRAL_BINS], out: &mut [f32; 256]) {
        // 1. Pre-twiddle + bit-reversed scatter into 64-point complex buffer.
        let mut re = [0.0f32; 64];
        let mut im = [0.0f32; 64];
        for k in 0..64usize {
            let x_even = input[2 * k];
            let x_odd = input[127 - 2 * k];
            let c = self.twiddle_cos[k];
            let s = self.twiddle_sin[k];
            // (x_even + i * x_odd) * (c - i * s)
            let dst = self.bitrev[k] as usize;
            re[dst] = x_even * c + x_odd * s;
            im[dst] = x_odd * c - x_even * s;
        }

        // 2. 6-stage in-place radix-2 DIT complex DFT (64 points, e^{-i 2pi k m / 64}).
        let mut half = 1usize;
        let mut step = 32usize;
        while half < 64 {
            let len = half << 1;
            let mut base = 0usize;
            while base < 64 {
                let mut tw = 0usize;
                for j in 0..half {
                    let u = base + j;
                    let v = u + half;
                    let wr = self.fft_cos[tw];
                    let wi = -self.fft_sin[tw];
                    let tr = re[v] * wr - im[v] * wi;
                    let ti = re[v] * wi + im[v] * wr;
                    re[v] = re[u] - tr;
                    im[v] = im[u] - ti;
                    re[u] += tr;
                    im[u] += ti;
                    tw += step;
                }
                base += len;
            }
            half = len;
            step >>= 1;
        }

        // 3. Post-twiddle into 128-point DCT-IV buffer c[0..128].
        let mut dct4 = [0.0f32; 128];
        for m in 0..64usize {
            let c = self.twiddle_cos[m];
            let s = self.twiddle_sin[m];
            // U[m] = W[m] * (c - i * s)
            let ur = re[m] * c + im[m] * s;
            let ui = im[m] * c - re[m] * s;
            dct4[2 * m] = ur;
            dct4[127 - 2 * m] = -ui;
        }

        // 4. Quarter-wave IMDCT symmetry expansion + scaling (1/64) + sine window.
        const SCALE: f32 = 1.0 / 64.0;
        for n in 0..64usize {
            out[n] = dct4[n + 64] * SCALE * self.window[n];
            out[64 + n] = -dct4[127 - n] * SCALE * self.window[64 + n];
            out[128 + n] = -dct4[63 - n] * SCALE * self.window[128 + n];
            out[192 + n] = -dct4[n] * SCALE * self.window[192 + n];
        }
    }
}

/// Fast 128-point IMDCT: 128 spectral bins -> 256 windowed samples.
///
/// Uses the O(N log N) 64-point FFT decomposition with the caller's `window`.
pub fn imdct(input: &[f32; SPECTRAL_BINS], window: &[f32; 256], out: &mut [f32; 256]) {
    let mut tables = ImdctTables::new();
    tables.window = *window;
    tables.imdct(input, out);
}

/// Forward MDCT (exact Princen-Bradley analysis transform paired with [`imdct`]):
/// 256 samples -> 128 spectral bins. Used by [`HcaEncoder`].
#[allow(clippy::needless_range_loop)]
pub fn mdct(input: &[f32; 256], window: &[f32; 256], out: &mut [f32; SPECTRAL_BINS]) {
    for k in 0..SPECTRAL_BINS {
        let mut sum = 0.0f32;
        for n in 0..256usize {
            let ang = (math::PI / 128.0) * (n as f32 + 64.5) * (k as f32 + 0.5);
            sum += input[n] * window[n] * math::cos_f32(ang);
        }
        out[k] = sum;
    }
}

/// Clean-room per-band resolution calculation from 6-bit scalefactor, ATH, `a`, and `b`.
fn band_resolution(sf: u8, ath: u8, a: u16, b: u8, band: usize, min_res: u8, max_res: u8) -> u8 {
    if sf == 0 {
        return 0;
    }
    let noise_ref = if band < b as usize {
        a as i32
    } else {
        (a as i32).saturating_sub(1)
    };
    let eff_sf = sf as i32 - ((ath as i32 + noise_ref) >> 3);
    eff_sf.clamp(min_res as i32, max_res as i32).max(0) as u8
}

/// Clean-room logarithmic dequantizer gain for a 6-bit scalefactor (`0..=63`).
fn dequant_scale(sf: u8, res: u8) -> f32 {
    if sf == 0 || res == 0 {
        return 0.0;
    }
    let max_mag = ((1u32 << (res - 1)).saturating_sub(1)).max(1) as f32;
    math::pow2_f32((sf as f32 - 36.0) / 3.0) / max_mag
}

/// Streaming HCA decoder over a borrowed stream.
#[derive(Debug, Clone)]
pub struct HcaDecoder<'a> {
    view: HcaView<'a>,
    cipher: HcaCipher,
    imdct_tables: ImdctTables,
    ath_table: [u8; SPECTRAL_BINS],
    overlap: [[f32; SAMPLES_PER_SUBFRAME]; MAX_CHANNELS],
    frame_index: usize,
    looping: bool,
    pending: [f32; SAMPLES_PER_FRAME * MAX_CHANNELS],
    pending_frames: usize,
    pending_pos: usize,
    finished: bool,
    scratch: [u8; MAX_FRAME_SIZE],
}

impl<'a> HcaDecoder<'a> {
    /// Create a decoder from a complete HCA stream.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Self::with_key(bytes, 0)
    }

    /// Create a decoder with a cipher key (type 56) and reset.
    pub fn with_key(bytes: &'a [u8], key: u64) -> Result<Self> {
        Self::with_key_and_subkey(bytes, key, 0)
    }

    /// Create a decoder with a cipher key and 16-bit AWB subkey (type 56).
    pub fn with_key_and_subkey(bytes: &'a [u8], key: u64, subkey: u16) -> Result<Self> {
        let view = HcaView::parse(bytes)?;
        if view.header.channels as usize > MAX_CHANNELS {
            return Err(Error::InvalidField {
                what: "channels",
                value: view.header.channels as u64,
            });
        }
        let cipher = HcaCipher::for_header_with_subkey(&view.header, key, subkey);
        let ath_table = build_ath_table(view.header.effective_ath_type(), view.header.sample_rate);
        Ok(Self {
            view,
            cipher,
            imdct_tables: ImdctTables::new(),
            ath_table,
            overlap: [[0.0; SAMPLES_PER_SUBFRAME]; MAX_CHANNELS],
            frame_index: 0,
            looping: false,
            pending: [0.0; SAMPLES_PER_FRAME * MAX_CHANNELS],
            pending_frames: 0,
            pending_pos: 0,
            finished: false,
            scratch: [0u8; MAX_FRAME_SIZE],
        })
    }

    /// Parsed header.
    pub fn header(&self) -> &HcaHeader {
        &self.view.header
    }

    /// Channel count.
    pub fn channels(&self) -> usize {
        self.view.header.channels as usize
    }

    /// Enable seamless looping over the header loop frames.
    pub fn set_looping(&mut self, on: bool) {
        self.looping = on;
    }

    /// Reset to the first frame, clearing overlap state.
    pub fn reset(&mut self) {
        self.frame_index = 0;
        self.overlap = [[0.0; SAMPLES_PER_SUBFRAME]; MAX_CHANNELS];
        self.pending_frames = 0;
        self.pending_pos = 0;
        self.finished = false;
    }

    /// Parse and validate a single frame's CRC and header (`0xFFFF` sync, `a`, `b`,
    /// delta-coded 6-bit scalefactors, intensity, HFR) without advancing decoder state.
    pub fn inspect_frame_header(&self, frame_index: usize) -> Result<HcaFrameHeader> {
        let frame = self.view.frame(frame_index).ok_or(Error::UnexpectedEof)?;
        let flen = frame.len();
        let mut buf = [0u8; MAX_FRAME_SIZE];
        buf[..flen].copy_from_slice(frame);
        self.cipher.apply_inverse(&mut buf[..flen]);
        if crc16(&buf[..flen]) != 0 {
            let stored = u16::from_be_bytes([buf[flen - 2], buf[flen - 1]]);
            return Err(Error::CrcMismatch {
                stored,
                computed: crc16(&buf[..flen - 2]),
            });
        }
        HcaFrameHeader::parse(&buf[..flen - 2], &self.view.header)
    }

    #[allow(clippy::needless_range_loop)]
    fn decode_next_frame(&mut self) -> Result<bool> {
        let total = self.view.header.frame_count as usize;
        if self.frame_index >= total {
            if self.looping {
                if let Some(lp) = self.view.header.looping {
                    self.frame_index = lp.start_frame as usize;
                } else {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        let frame = self
            .view
            .frame(self.frame_index)
            .ok_or(Error::UnexpectedEof)?;
        let flen = frame.len();
        self.scratch[..flen].copy_from_slice(frame);
        self.cipher.apply_inverse(&mut self.scratch[..flen]);
        if crc16(&self.scratch[..flen]) != 0 {
            let stored = u16::from_be_bytes([self.scratch[flen - 2], self.scratch[flen - 1]]);
            return Err(Error::CrcMismatch {
                stored,
                computed: crc16(&self.scratch[..flen - 2]),
            });
        }
        self.frame_index += 1;

        let header = &self.view.header;
        let ch = header.channels as usize;
        let (base_bands, stereo_bands) = header.band_layout();
        let coded_primary = (base_bands + stereo_bands).min(SPECTRAL_BINS);
        let ch_types = header.channel_types();
        let (min_res, max_res) = header.resolution_range();
        let payload = &self.scratch[..flen - 2];

        let fhdr = HcaFrameHeader::parse(payload, header)?;

        // Precompute per-channel, per-bin resolution and dequantization step.
        let mut resolutions = [[0u8; SPECTRAL_BINS]; MAX_CHANNELS];
        let mut steps = [[0.0f32; SPECTRAL_BINS]; MAX_CHANNELS];
        for c in 0..ch {
            let count = match ch_types[c] {
                HcaChannelType::StereoSecondary => base_bands,
                _ => coded_primary,
            };
            for b in 0..count {
                let sf = fhdr.scale_factors[c][b];
                let res = band_resolution(
                    sf,
                    self.ath_table[b],
                    fhdr.acceptable_noise_level,
                    fhdr.evaluation_boundary,
                    b,
                    min_res,
                    max_res,
                );
                resolutions[c][b] = res;
                steps[c][b] = dequant_scale(sf, res);
            }
        }

        let mut br = BitReader::new(payload);
        // Advance to exact start of subframe 0 spectral data.
        let mut skip = fhdr.header_bits;
        while skip > 0 {
            let take = skip.min(32) as u8;
            let _ = br.read(take)?;
            skip -= take as usize;
        }

        let mut block = [0.0f32; 256];
        for sf_idx in 0..SUBFRAMES_PER_FRAME {
            let mut spectra = [[0.0f32; SPECTRAL_BINS]; MAX_CHANNELS];
            for c in 0..ch {
                let count = match ch_types[c] {
                    HcaChannelType::StereoSecondary => base_bands,
                    _ => coded_primary,
                };
                for b in 0..count {
                    let res = resolutions[c][b];
                    if res == 0 {
                        continue;
                    }
                    if br.remaining() < res as usize {
                        // Graceful zero-fill when a stream's spectral payload ends early.
                        break;
                    }
                    let q = br.read_signed(res)?;
                    spectra[c][b] = q as f32 * steps[c][b];
                }
            }
            // Intensity stereo coupling for paired channels over base_bands..coded_primary.
            if stereo_bands > 0 {
                for c in 0..ch {
                    if ch_types[c] == HcaChannelType::StereoPrimary && c + 1 < ch {
                        let iv = fhdr.intensity[c + 1][sf_idx];
                        let ratio_l = (iv as f32) / 14.0;
                        let ratio_r = 1.0 - ratio_l;
                        for b in base_bands..coded_primary {
                            let s = spectra[c][b];
                            spectra[c][b] = s * ratio_l;
                            spectra[c + 1][b] = s * ratio_r;
                        }
                    }
                }
            }
            // Fast 64-point FFT-based 128-point IMDCT + overlap-add per channel.
            let sample_base = sf_idx * SAMPLES_PER_SUBFRAME;
            for c in 0..ch {
                self.imdct_tables.imdct(&spectra[c], &mut block);
                for i in 0..SAMPLES_PER_SUBFRAME {
                    self.pending[(sample_base + i) * ch + c] = self.overlap[c][i] + block[i];
                }
                self.overlap[c].copy_from_slice(&block[SAMPLES_PER_SUBFRAME..]);
            }
        }

        self.pending_frames = SAMPLES_PER_FRAME;
        self.pending_pos = 0;
        Ok(true)
    }

    /// Decode interleaved `i16` PCM. Returns sample-frames per channel.
    pub fn decode_i16(&mut self, out: &mut [i16]) -> Result<usize> {
        self.decode_f32_planar_into_i16(out)
    }

    #[allow(clippy::collapsible_if)]
    fn decode_f32_planar_into_i16(&mut self, out: &mut [i16]) -> Result<usize> {
        if self.finished {
            return Ok(0);
        }
        let ch = self.channels();
        if ch == 0 || out.len() < ch {
            return Err(Error::BufferTooSmall {
                needed: ch,
                actual: out.len(),
            });
        }
        let capacity = out.len() / ch;
        let mut written = 0usize;
        while written < capacity {
            if self.pending_pos >= self.pending_frames {
                if !self.decode_next_frame()? {
                    self.finished = true;
                    break;
                }
            }
            let avail = self.pending_frames - self.pending_pos;
            let take = avail.min(capacity - written);
            for i in 0..take {
                let src = (self.pending_pos + i) * ch;
                for c in 0..ch {
                    out[(written + i) * ch + c] = clamp16(self.pending[src + c] * 32767.0);
                }
            }
            self.pending_pos += take;
            written += take;
        }
        Ok(written)
    }

    /// Decode planar `f32` PCM (`planes[c]` per channel). Returns frames.
    pub fn decode_f32_planar(&mut self, planes: &mut [&mut [f32]]) -> Result<usize> {
        let ch = self.channels();
        if planes.len() < ch {
            return Err(Error::BufferTooSmall {
                needed: ch,
                actual: planes.len(),
            });
        }
        let capacity = planes.iter().take(ch).map(|p| p.len()).min().unwrap_or(0);
        let mut scratch = [0i16; 128 * MAX_CHANNELS];
        let want = capacity.min(128);
        if want == 0 {
            return Ok(0);
        }
        let got = self.decode_i16(&mut scratch[..want * ch])?;
        for c in 0..ch {
            for i in 0..got {
                planes[c][i] = scratch[i * ch + c] as f32 / 32767.0;
            }
        }
        Ok(got)
    }

    /// Decode interleaved `f32` PCM. Returns frames.
    pub fn decode_f32(&mut self, out: &mut [f32]) -> Result<usize> {
        let ch = self.channels();
        let capacity = out.len().checked_div(ch).unwrap_or(0);
        let mut scratch = [0i16; 128 * MAX_CHANNELS];
        let want = capacity.min(128);
        if want == 0 {
            return Ok(0);
        }
        let got = self.decode_i16(&mut scratch[..want * ch])?;
        for i in 0..got * ch {
            out[i] = scratch[i] as f32 / 32767.0;
        }
        Ok(got)
    }
}

/// Stateful clean-room HCA encoder producing real-framed 8-subframe (1024-sample)
/// HCA frames (`0xFFFF` sync, `a`/`b`, delta-coded 6-bit scalefactors, intensity,
/// spectral data, CRC-16/BUYPASS, and cipher).
#[derive(Debug, Clone)]
pub struct HcaEncoder {
    /// Channels.
    pub channels: u8,
    /// Total bands (up to 128).
    pub bands: u8,
    /// Stereo bands.
    pub stereo_bands: u8,
    /// Minimum resolution.
    pub min_resolution: u8,
    /// Maximum resolution.
    pub max_resolution: u8,
    /// ATH type.
    pub ath_type: u16,
    /// Sample rate (used for ATH table construction).
    pub sample_rate: u32,
    /// Frame size in bytes.
    pub frame_size: u16,
    cipher: HcaCipher,
    window: [f32; 256],
}

impl HcaEncoder {
    /// Create an encoder.
    pub fn new(
        channels: u8,
        bands: u8,
        stereo_bands: u8,
        frame_size: u16,
        cipher: HcaCipher,
    ) -> Result<Self> {
        if channels == 0
            || channels as usize > MAX_CHANNELS
            || bands == 0
            || bands as usize > SPECTRAL_BINS
        {
            return Err(Error::InvalidParameter("channels/bands"));
        }
        if (frame_size as usize) < 16 || frame_size as usize > MAX_FRAME_SIZE {
            return Err(Error::InvalidParameter("frame_size"));
        }
        Ok(Self {
            channels,
            bands,
            stereo_bands,
            min_resolution: 1,
            max_resolution: 15,
            ath_type: 0,
            sample_rate: 44100,
            frame_size,
            cipher,
            window: sine_window(),
        })
    }

    /// Encode one 8-subframe HCA frame by repeating the supplied per-channel spectra
    /// across all 8 subframes (convenience for synthetic signal tests & benchmarks).
    #[allow(clippy::needless_range_loop)]
    pub fn encode_frame_spectrum(
        &self,
        spectra: &[&[f32; SPECTRAL_BINS]],
        out: &mut [u8],
    ) -> Result<usize> {
        let flen = self.frame_size as usize;
        if out.len() < flen {
            return Err(Error::BufferTooSmall {
                needed: flen,
                actual: out.len(),
            });
        }
        let ch = self.channels as usize;
        if spectra.len() < ch {
            return Err(Error::BufferTooSmall {
                needed: ch,
                actual: spectra.len(),
            });
        }
        let total_bands = self.bands as usize;
        let base_bands = total_bands.saturating_sub(self.stereo_bands as usize);
        let ath_table = build_ath_table(self.ath_type, self.sample_rate);

        // Choose a 6-bit scalefactor (0..=63) per (channel, bin).
        let mut sfs = [[0u8; SPECTRAL_BINS]; MAX_CHANNELS];
        for c in 0..ch {
            let count = if self.stereo_bands > 0 && c == 1 {
                base_bands
            } else {
                total_bands
            };
            for b in 0..count {
                sfs[c][b] = choose_scalefactor(spectra[c][b].abs());
            }
        }

        // Choose acceptable_noise_level `a` so the 8 subframes fit inside `flen - 2`.
        let max_bits = (flen - 2) * 8;
        let mut a = 0u16;
        let b_bound = 0u8;
        let bw = loop {
            let mut bw = BitWriter::new();
            bw.write(FRAME_SYNC as u32, 16)?;
            bw.write(a as u32, 9)?;
            bw.write(b_bound as u32, 7)?;
            for c in 0..ch {
                let count = if self.stereo_bands > 0 && c == 1 {
                    base_bands
                } else {
                    total_bands
                };
                if sfs[c][..count].iter().all(|&s| s == 0) {
                    bw.write(0, 3)?; // delta_bits = 0 (all zeros)
                } else {
                    bw.write(6, 3)?; // delta_bits = 6 (raw 6-bit scalefactors)
                    for b in 0..count {
                        bw.write(sfs[c][b] as u32, 6)?;
                    }
                }
                if self.stereo_bands > 0 && c == 1 {
                    for _ in 0..SUBFRAMES_PER_FRAME {
                        bw.write(7, 4)?; // neutral intensity
                    }
                }
            }
            let mut fit = true;
            for _sf_idx in 0..SUBFRAMES_PER_FRAME {
                for c in 0..ch {
                    let count = if self.stereo_bands > 0 && c == 1 {
                        base_bands
                    } else {
                        total_bands
                    };
                    for b in 0..count {
                        let sf = sfs[c][b];
                        let res = band_resolution(
                            sf,
                            ath_table[b],
                            a,
                            b_bound,
                            b,
                            self.min_resolution,
                            self.max_resolution,
                        );
                        if res == 0 {
                            continue;
                        }
                        if bw.bit_pos + res as usize > max_bits {
                            fit = false;
                            break;
                        }
                        let step = dequant_scale(sf, res);
                        let q = if step == 0.0 {
                            0
                        } else {
                            let v = spectra[c][b] / step;
                            (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
                        };
                        let lim = (1i32 << (res - 1)) - 1;
                        bw.write_signed(q.clamp(-lim, lim), res)?;
                    }
                    if !fit {
                        break;
                    }
                }
                if !fit {
                    break;
                }
            }
            if fit || a >= 504 {
                if !fit {
                    return Err(Error::BufferTooSmall {
                        needed: bw.byte_len() + 2,
                        actual: flen,
                    });
                }
                break bw;
            }
            a = (a + 8).min(511);
        };

        let payload_len = bw.byte_len();
        if payload_len + 2 > flen {
            return Err(Error::BufferTooSmall {
                needed: payload_len + 2,
                actual: flen,
            });
        }
        out[..flen].fill(0);
        out[..payload_len].copy_from_slice(&bw.data[..payload_len]);
        let crc = crc16(&out[..flen - 2]);
        out[flen - 2..flen].copy_from_slice(&crc.to_be_bytes());
        self.cipher.apply(&mut out[..flen]);
        Ok(flen)
    }

    /// Encode one frame from 256 windowed PCM samples per channel (MDCT).
    pub fn encode_frame_pcm(&self, pcm: &[&[f32; 256]], out: &mut [u8]) -> Result<usize> {
        let ch = self.channels as usize;
        if pcm.len() < ch {
            return Err(Error::BufferTooSmall {
                needed: ch,
                actual: pcm.len(),
            });
        }
        let mut spectra = [[0.0f32; SPECTRAL_BINS]; MAX_CHANNELS];
        for c in 0..ch {
            mdct(pcm[c], &self.window, &mut spectra[c]);
        }
        let refs: [&[f32; SPECTRAL_BINS]; MAX_CHANNELS] = [
            &spectra[0],
            &spectra[1],
            &spectra[2],
            &spectra[3],
            &spectra[4],
            &spectra[5],
            &spectra[6],
            &spectra[7],
        ];
        self.encode_frame_spectrum(&refs[..ch], out)
    }
}

fn choose_scalefactor(abs_val: f32) -> u8 {
    if abs_val <= 1e-9 {
        return 0;
    }
    for sf in 1..=63u8 {
        if abs_val <= math::pow2_f32((sf as f32 - 36.0) / 3.0) {
            return sf;
        }
    }
    63
}

/// Build a synthetic HCA header (`fmt` + `comp` + optional `ath`/`loop`/`ciph`/`rva` + CRC-16).
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
pub fn build_header(
    version: u16,
    channels: u8,
    sample_rate: u32,
    frame_count: u32,
    frame_size: u16,
    comp: HcaComp,
    ath_type: Option<u16>,
    looping: Option<HcaLoop>,
    cipher_type: u16,
    rva_volume: Option<f32>,
) -> Result<alloc::vec::Vec<u8>> {
    use alloc::vec;
    let mut body = vec![0u8; 8];
    body[0..4].copy_from_slice(b"HCA\0");
    body[4..6].copy_from_slice(&version.to_be_bytes());
    let mut fmt = vec![0u8; 16];
    fmt[0..4].copy_from_slice(b"fmt\0");
    fmt[4] = channels;
    fmt[5] = (sample_rate >> 16) as u8;
    fmt[6] = (sample_rate >> 8) as u8;
    fmt[7] = sample_rate as u8;
    fmt[8..12].copy_from_slice(&frame_count.to_be_bytes());
    body.extend_from_slice(&fmt);
    let mut compb = vec![0u8; 16];
    compb[0..4].copy_from_slice(b"comp");
    compb[4..6].copy_from_slice(&frame_size.to_be_bytes());
    compb[6] = comp.min_resolution;
    compb[7] = comp.max_resolution;
    compb[8] = comp.track_count;
    compb[9] = comp.channel_config;
    compb[10] = comp.total_bands;
    compb[11] = comp.base_bands;
    compb[12] = comp.stereo_bands;
    compb[13] = comp.bands_per_hfr_group;
    body.extend_from_slice(&compb);
    if let Some(a) = ath_type {
        let mut ath = vec![0u8; 6];
        ath[0..4].copy_from_slice(b"ath\0");
        ath[4..6].copy_from_slice(&a.to_be_bytes());
        body.extend_from_slice(&ath);
    }
    if let Some(lp) = looping {
        let mut lp_b = vec![0u8; 16];
        lp_b[0..4].copy_from_slice(b"loop");
        lp_b[4..8].copy_from_slice(&lp.start_frame.to_be_bytes());
        lp_b[8..12].copy_from_slice(&lp.end_frame.to_be_bytes());
        lp_b[12..14].copy_from_slice(&lp.pre_samples.to_be_bytes());
        lp_b[14..16].copy_from_slice(&lp.post_samples.to_be_bytes());
        body.extend_from_slice(&lp_b);
    }
    if cipher_type != 0 {
        let mut c = vec![0u8; 6];
        c[0..4].copy_from_slice(b"ciph");
        c[4..6].copy_from_slice(&cipher_type.to_be_bytes());
        body.extend_from_slice(&c);
    }
    if let Some(v) = rva_volume {
        let mut r = vec![0u8; 8];
        r[0..4].copy_from_slice(b"rva\0");
        r[4..8].copy_from_slice(&v.to_bits().to_be_bytes());
        body.extend_from_slice(&r);
    }
    body.extend_from_slice(&[0u8; 2]); // CRC placeholder
    let header_size = body.len() as u16;
    body[6..8].copy_from_slice(&header_size.to_be_bytes());
    let crc = crc16(&body[..body.len() - 2]);
    let n = body.len();
    body[n - 2..].copy_from_slice(&crc.to_be_bytes());
    debug_assert_eq!(crc16(&body), 0);
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_signed() {
        let mut br = BitReader::new(&[0b1011_0000]);
        assert_eq!(br.read(2).unwrap(), 0b10);
        assert_eq!(br.read_signed(2).unwrap(), -1);
    }

    #[test]
    fn cipher_roundtrip_and_anchors() {
        for c in [
            HcaCipher::type1(),
            HcaCipher::type56(0x1234_5678_9ABC_DEF0),
            HcaCipher::type56_with_awb_subkey(0x1234_5678_9ABC_DEF0, 0x1A2B),
        ] {
            // Sync anchors 0x00 and 0xFF must remain fixed.
            assert_eq!(c.table_entry(0x00), 0x00);
            assert_eq!(c.table_entry(0xFF), 0xFF);
            let orig = [0xFFu8, 0xFF, 0x00, 1, 42, 200, 7];
            let mut data = orig;
            c.apply(&mut data);
            assert_eq!(data[0], 0xFF);
            assert_eq!(data[1], 0xFF);
            assert_eq!(data[2], 0x00);
            assert_ne!(&data[3..], &orig[3..]);
            c.apply_inverse(&mut data);
            assert_eq!(data, orig);
        }
    }

    #[test]
    fn fast_fft_imdct_matches_naive_reference() {
        let tables = ImdctTables::new();
        let mut spec = [0.0f32; SPECTRAL_BINS];
        for (k, v) in spec.iter_mut().enumerate() {
            *v = math::sin_f32(k as f32 * 0.37 + 0.2) * (1.0 + (k % 7) as f32 * 0.15);
        }
        let mut fast_out = [0.0f32; 256];
        tables.imdct(&spec, &mut fast_out);

        // Direct O(N^2) mathematical reference for verification.
        let mut ref_out = [0.0f32; 256];
        for (n, dst) in ref_out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &x) in spec.iter().enumerate() {
                let ang = (math::PI / 128.0) * (n as f32 + 64.5) * (k as f32 + 0.5);
                sum += x * math::cos_f32(ang);
            }
            *dst = sum * (1.0 / 64.0) * tables.window[n];
        }

        let mut max_err = 0.0f32;
        for i in 0..256 {
            let e = (fast_out[i] - ref_out[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-4, "FFT IMDCT max error vs naive: {max_err}");
    }

    #[test]
    fn mdct_imdct_tdac_perfect_reconstruction() {
        let tables = ImdctTables::new();
        // Synthesize 384 samples (2 overlapping 256-sample blocks -> middle 128 samples exact).
        let mut pcm = [0.0f32; 384];
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = 0.6 * math::sin_f32(i as f32 * 0.13) + 0.3 * math::cos_f32(i as f32 * 0.41);
        }
        let mut b0 = [0.0f32; 256];
        let mut b1 = [0.0f32; 256];
        b0.copy_from_slice(&pcm[0..256]);
        b1.copy_from_slice(&pcm[128..384]);

        let mut s0 = [0.0f32; 128];
        let mut s1 = [0.0f32; 128];
        mdct(&b0, &tables.window, &mut s0);
        mdct(&b1, &tables.window, &mut s1);

        let mut o0 = [0.0f32; 256];
        let mut o1 = [0.0f32; 256];
        tables.imdct(&s0, &mut o0);
        tables.imdct(&s1, &mut o1);

        let mut max_err = 0.0f32;
        for i in 0..128 {
            let recon = o0[128 + i] + o1[i];
            let err = (recon - pcm[128 + i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-4, "TDAC reconstruction error: {max_err}");
    }

    #[test]
    fn imdct_zero_and_impulse() {
        let tables = ImdctTables::new();
        let mut out = [0.0f32; 256];
        tables.imdct(&[0.0; 128], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut imp = [0.0f32; 128];
        imp[0] = 64.0;
        tables.imdct(&imp, &mut out);
        assert!(out.iter().any(|&v| v != 0.0));
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
