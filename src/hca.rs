//! CRI HCA (v1.3 / v2.0 / v3.0) header parsing, cipher and frame decoding.
//!
//! The header starts with `HCA\0` (tags may be masked with `0x7F` per byte),
//! a big-endian `u16` version and `u16` header size, followed by chunks
//! (`fmt`, `comp`, `dec`, `vbr`, `ath`, `loop`, `ciph`, `rva`, `comm`,
//! `pad`) and a trailing CRC-16/BUYPASS over the whole header.
//!
//! Frame decoding in this clean-room implementation is deterministic:
//! each frame (after cipher removal and CRC-16 validation) is read with an
//! MSB-first bit reader. Per channel, 4-bit scalefactors are unpacked for
//! every band, intensity values are unpacked for stereo bands, then each
//! of the 128 spectral bins is unpacked with a bit resolution derived from
//! its scalefactor minus its ATH attenuation (clamped to the header's
//! resolution range) and dequantised by `2^(scalefactor-8)`. The spectrum
//! is transformed with an IMDCT (128 bins -> 256 samples, sine window)
//! and overlap-added with a 128-sample hop to PCM.

use crate::crc::crc16;
use crate::math;
use crate::{Error, Result};

/// Spectral bins per HCA frame per channel in this implementation.
pub const SPECTRAL_BINS: usize = 128;
/// Samples produced per frame per channel (IMDCT hop).
pub const SAMPLES_PER_FRAME: usize = 128;
/// Maximum channels supported without per-frame allocation.
pub const MAX_CHANNELS: usize = 8;
/// Maximum accepted frame size.
pub const MAX_FRAME_SIZE: usize = 4096;

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
    /// Total number of scalefactor bands.
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
    /// Channel count (1..=8 for decoding; header allows up to 16).
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
    /// ATH curve type, if present.
    pub ath_type: Option<u16>,
    /// VBR maximum frame size field, if a `vbr` chunk was present.
    pub vbr: Option<u16>,
    /// Loop points, if present.
    pub looping: Option<HcaLoop>,
    /// Cipher type: 0 none, 1 keyless, 56 keyed.
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

    /// Parse an HCA header from the start of `bytes` and validate its CRC.
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
        // CRC validation: a well-formed header checksums to zero.
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
        }
        Ok(h)
    }

    /// Total samples per channel declared (`frame_count * 128`).
    pub fn total_samples(&self) -> u64 {
        self.frame_count as u64 * SAMPLES_PER_FRAME as u64
    }

    /// Byte length of the frame stream.
    pub fn data_len(&self) -> usize {
        self.frame_count as usize * self.frame_size as usize
    }

    /// Effective band count (defaults to 32 without a `comp` chunk).
    pub fn bands(&self) -> usize {
        match self.comp {
            Some(c) if c.total_bands > 0 => c.total_bands as usize,
            _ => 32,
        }
    }

    /// Resolution range (defaults to `1..=15`).
    pub fn resolution_range(&self) -> (u8, u8) {
        match self.comp {
            Some(c) => (c.min_resolution, c.max_resolution),
            None => (1, 15),
        }
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

/// HCA cipher: byte substitution table (types 0, 1 and 56).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcaCipher {
    table: [u8; 256],
    inverse: [u8; 256],
    /// Cipher type this table was built for.
    pub cipher_type: u16,
}

fn cipher_seed(cipher_type: u16, key: u64, subkey: u64) -> u32 {
    let mut s = match cipher_type {
        1 => 0x5A3C_9E77u32,
        _ => (key as u32) ^ ((key >> 32) as u32) ^ (subkey as u32) ^ ((subkey >> 32) as u32),
    };
    if s == 0 {
        s = 0x1357_9BDF;
    }
    s
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

    /// Keyless cipher (type 1): fixed-seed permutation table.
    pub fn type1() -> Self {
        Self::build(1, 0, 0)
    }

    /// Keyed cipher (type 56) from a 64-bit key; subkey derived from the key.
    pub fn type56(key: u64) -> Self {
        let subkey = ((key >> 16) & 0xFFFF_FFFF) ^ 0xA5A5_5A5A;
        Self::build(56, key, subkey)
    }

    /// Keyed cipher (type 56) from a 64-bit key and an explicit subkey.
    pub fn type56_with_subkey(key: u64, subkey: u64) -> Self {
        Self::build(56, key, subkey)
    }

    fn build(cipher_type: u16, key: u64, subkey: u64) -> Self {
        let mut table = [0u8; 256];
        for (i, t) in table.iter_mut().enumerate() {
            *t = i as u8;
        }
        // Fisher-Yates shuffle driven by a Numerical-Recipes LCG.
        let mut state = cipher_seed(cipher_type, key, subkey);
        let mut next = move || {
            state = state.wrapping_mul(0x0019_660D).wrapping_add(0x3C6E_F35F);
            state >> 16
        };
        for i in (1..256usize).rev() {
            let j = (next() as usize) % (i + 1);
            table.swap(i, j);
        }
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
        match header.cipher_type {
            1 => Self::type1(),
            56 => Self::type56(key),
            _ => Self::none(),
        }
    }

    /// Encrypt (forward substitution) in place.
    pub fn apply(&self, data: &mut [u8]) {
        for b in data.iter_mut() {
            *b = self.table[*b as usize];
        }
    }

    /// Decrypt (inverse substitution) in place.
    pub fn apply_inverse(&self, data: &mut [u8]) {
        for b in data.iter_mut() {
            *b = self.inverse[*b as usize];
        }
    }

    /// Forward table entry.
    pub fn table_entry(&self, b: u8) -> u8 {
        self.table[b as usize]
    }
}

/// ATH attenuation (in resolution bits) for a spectral bin.
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

    /// Bits remaining.
    pub fn remaining(&self) -> usize {
        self.data.len() * 8 - self.bit_pos.min(self.data.len() * 8)
    }

    /// Read `n` (<= 32) bits as an unsigned value.
    pub fn read(&mut self, n: u8) -> Result<u32> {
        if n > 32 {
            return Err(Error::InvalidParameter("bit width"));
        }
        let mut v: u32 = 0;
        for _ in 0..n {
            let byte = self
                .data
                .get(self.bit_pos / 8)
                .ok_or(Error::BitstreamDesync)?;
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.bit_pos += 1;
        }
        Ok(v)
    }

    /// Read `n` bits as a sign-extended signed value (`n == 0` yields 0).
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

fn clamp16(v: f32) -> i16 {
    if v >= 32767.0 {
        32767
    } else if v <= -32768.0 {
        -32768
    } else {
        v as i16
    }
}

/// Sine window of length 256 used by the IMDCT.
pub fn sine_window() -> [f32; 256] {
    let mut w = [0f32; 256];
    for (n, x) in w.iter_mut().enumerate() {
        *x = math::sin_f32(math::PI * (n as f32 + 0.5) / 256.0);
    }
    w
}

/// Inverse MDCT: 128 spectral bins -> 256 windowed samples.
///
/// `out[n] = (1/64) * sum_k X[k] * cos((pi/128)*(n+0.5+32)*(k+0.5))`,
/// then multiplied by the sine window. Computed with a cosine recurrence
/// (no allocation, deterministic).
pub fn imdct(input: &[f32; SPECTRAL_BINS], window: &[f32; 256], out: &mut [f32; 256]) {
    for n in 0..256usize {
        let phase = (math::PI / 128.0) * (n as f32 + 0.5 + 32.0);
        // cos(phase*(k+0.5)) via rotation recurrence over k.
        let mut c = math::cos_f32(phase * 0.5);
        let mut s = math::sin_f32(phase * 0.5);
        let step_c = math::cos_f32(phase);
        let step_s = math::sin_f32(phase);
        let mut sum = 0.0f32;
        for &x in input.iter() {
            sum += x * c;
            let nc = c * step_c - s * step_s;
            let ns = s * step_c + c * step_s;
            c = nc;
            s = ns;
        }
        out[n] = sum * (1.0 / 64.0) * window[n];
    }
}

/// Forward MDCT (transpose of [`imdct`] without the window scale factor):
/// 256 windowed samples -> 128 spectral bins. Used by [`HcaEncoder`].
#[allow(clippy::needless_range_loop)]
pub fn mdct(input: &[f32; 256], window: &[f32; 256], out: &mut [f32; SPECTRAL_BINS]) {
    for k in 0..SPECTRAL_BINS {
        let mut sum = 0.0f32;
        for n in 0..256usize {
            let ang = (math::PI / 128.0) * (n as f32 + 0.5 + 32.0) * (k as f32 + 0.5);
            sum += input[n] * window[n] * math::cos_f32(ang);
        }
        out[k] = sum * (1.0 / 64.0);
    }
}

/// Streaming HCA decoder over a borrowed stream.
#[derive(Debug, Clone)]
pub struct HcaDecoder<'a> {
    view: HcaView<'a>,
    cipher: HcaCipher,
    window: [f32; 256],
    overlap: [[f32; SAMPLES_PER_FRAME]; MAX_CHANNELS],
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
        let view = HcaView::parse(bytes)?;
        if view.header.channels as usize > MAX_CHANNELS {
            return Err(Error::InvalidField {
                what: "channels",
                value: view.header.channels as u64,
            });
        }
        let cipher = HcaCipher::for_header(&view.header, key);
        Ok(Self {
            view,
            cipher,
            window: sine_window(),
            overlap: [[0.0; SAMPLES_PER_FRAME]; MAX_CHANNELS],
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
        self.overlap = [[0.0; SAMPLES_PER_FRAME]; MAX_CHANNELS];
        self.pending_frames = 0;
        self.pending_pos = 0;
        self.finished = false;
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
        let bands = header.bands().max(1);
        let (min_res, max_res) = header.resolution_range();
        let ath_type = header.ath_type.unwrap_or(0);
        let stereo_bands = header.comp.map(|c| c.stereo_bands as usize).unwrap_or(0);
        let payload_len = flen - 2;
        // SAFETY (logical): borrow payload immutably while writing spectrum
        // into local arrays; scratch is not mutated during unpacking.
        let mut scalefactors = [[0u8; 64]; MAX_CHANNELS];
        let mut spectrum = [[0.0f32; SPECTRAL_BINS]; MAX_CHANNELS];
        {
            let payload = &self.scratch[..payload_len];
            let mut br = BitReader::new(payload);
            for c in 0..ch {
                for b in 0..bands.min(64) {
                    scalefactors[c][b] = br.read(4)? as u8;
                }
            }
            // Intensity values for the second channel's stereo bands.
            let mut intensity = [8u8; 64];
            if ch >= 2 {
                for item in intensity.iter_mut().take(stereo_bands.min(64)) {
                    *item = br.read(4)? as u8;
                }
            }
            for bin in 0..SPECTRAL_BINS {
                let band = (bin * bands / SPECTRAL_BINS).min(bands - 1);
                for c in 0..ch {
                    let sf = scalefactors[c][band.min(63)];
                    let ath = ath_attenuation(ath_type, bin);
                    let res = (sf as i32 - ath as i32).clamp(min_res as i32, max_res as i32);
                    let res = if res < 0 { 0u8 } else { res as u8 };
                    let val = br.read_signed(res)?;
                    let mut coeff = val as f32 * math::pow2_i32(sf as i32 - 8);
                    if c == 1 && band >= bands.saturating_sub(stereo_bands) && stereo_bands > 0 {
                        let idx = band - bands.saturating_sub(stereo_bands);
                        coeff *= intensity[idx.min(63)] as f32 / 8.0;
                    }
                    spectrum[c][bin] = coeff;
                }
            }
        }
        // IMDCT + overlap-add per channel into the pending buffer.
        let mut block = [0.0f32; 256];
        for c in 0..ch {
            imdct(&spectrum[c], &self.window, &mut block);
            for i in 0..SAMPLES_PER_FRAME {
                self.pending[i * ch + c] = self.overlap[c][i] + block[i];
            }
            self.overlap[c].copy_from_slice(&block[SAMPLES_PER_FRAME..]);
        }
        self.pending_frames = SAMPLES_PER_FRAME;
        self.pending_pos = 0;
        Ok(true)
    }

    /// Decode interleaved `i16` PCM. Returns sample-frames per channel.
    pub fn decode_i16(&mut self, out: &mut [i16]) -> Result<usize> {
        let frames = self.decode_f32_planar_into_i16(out)?;
        Ok(frames)
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
        // Reuse the i16 path's frame advance, then rescale exactly.
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

/// Stateful HCA encoder producing frames in this crate's deterministic
/// frame format (scalefactors, intensity, spectral data, CRC, cipher).
#[derive(Debug, Clone)]
pub struct HcaEncoder {
    /// Channels.
    pub channels: u8,
    /// Bands.
    pub bands: u8,
    /// Stereo bands.
    pub stereo_bands: u8,
    /// Minimum resolution.
    pub min_resolution: u8,
    /// Maximum resolution.
    pub max_resolution: u8,
    /// ATH type.
    pub ath_type: u16,
    /// Frame size.
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
        if channels == 0 || channels as usize > MAX_CHANNELS || bands == 0 || bands > 64 {
            return Err(Error::InvalidParameter("channels/bands"));
        }
        if (frame_size as usize) < 16 || frame_size as usize > MAX_FRAME_SIZE {
            return Err(Error::InvalidParameter("frame_size"));
        }
        Ok(Self {
            channels,
            bands,
            stereo_bands,
            min_resolution: 0,
            max_resolution: 15,
            ath_type: 0,
            frame_size,
            cipher,
            window: sine_window(),
        })
    }

    /// Encode one frame directly from per-channel spectra.
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
        let bands = self.bands as usize;
        // Choose a scalefactor per (channel, band).
        let mut sfs = [[0u8; 64]; MAX_CHANNELS];
        for c in 0..ch {
            for b in 0..bands {
                let lo = b * SPECTRAL_BINS / bands;
                let hi = (b + 1) * SPECTRAL_BINS / bands;
                let mut max_abs = 0.0f32;
                for &v in &spectra[c][lo..hi] {
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                    }
                }
                sfs[c][b] = choose_scalefactor(max_abs, self);
            }
        }
        let mut bw = BitWriter::new();
        for c in 0..ch {
            for b in 0..bands {
                bw.write(sfs[c][b] as u32, 4)?;
            }
        }
        if ch >= 2 {
            for _ in 0..self.stereo_bands {
                bw.write(8, 4)?; // neutral intensity
            }
        }
        for bin in 0..SPECTRAL_BINS {
            let band = (bin * bands / SPECTRAL_BINS).min(bands - 1);
            for c in 0..ch {
                let sf = sfs[c][band];
                let ath = ath_attenuation(self.ath_type, bin);
                let res = (sf as i32 - ath as i32)
                    .clamp(self.min_resolution as i32, self.max_resolution as i32);
                let res = if res < 0 { 0u8 } else { res as u8 };
                let step = math::pow2_i32(sf as i32 - 8);
                let q = if step == 0.0 {
                    0
                } else {
                    (spectra[c][bin] / step + if spectra[c][bin] >= 0.0 { 0.5 } else { -0.5 })
                        as i32
                };
                let lim = if res == 0 { 0 } else { (1i32 << (res - 1)) - 1 };
                bw.write_signed(q.clamp(-lim - 1, lim), res)?;
            }
        }
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

fn choose_scalefactor(max_abs: f32, enc: &HcaEncoder) -> u8 {
    if max_abs <= 0.0 {
        return 0;
    }
    for sf in 0..=15u8 {
        let step = math::pow2_i32(sf as i32 - 8);
        let res = (sf as i32).clamp(enc.min_resolution as i32, enc.max_resolution as i32);
        let lim = if res <= 0 { 0 } else { (1i32 << (res - 1)) - 1 };
        if max_abs <= lim as f32 * step {
            return sf;
        }
    }
    15
}

/// Build a synthetic HCA header (fmt + comp + ath + optional loop/ciph/rva).
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
    // header size patched later
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
    fn cipher_roundtrip() {
        let c = HcaCipher::type56(0x1234_5678_9ABC_DEF0);
        let orig = [0u8, 1, 42, 200, 255, 7];
        let mut data = orig;
        c.apply(&mut data);
        assert_ne!(data, orig);
        c.apply_inverse(&mut data);
        assert_eq!(data, orig);
        let n = HcaCipher::none();
        let mut d2 = orig;
        n.apply(&mut d2);
        assert_eq!(d2, orig);
    }

    #[test]
    fn imdct_zero_and_impulse() {
        let w = sine_window();
        let mut out = [0.0f32; 256];
        imdct(&[0.0; 128], &w, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        let mut imp = [0.0f32; 128];
        imp[0] = 64.0;
        imdct(&imp, &w, &mut out);
        assert!(out.iter().any(|&v| v != 0.0));
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
