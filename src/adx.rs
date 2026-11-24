//! CRI ADX (v3 / v4 / v5) header parsing, ADPCM decoding and encryption.
//!
//! Header layout (big-endian, clean-room from observable framing):
//! `0x00: u16 0x8000`, `0x02: u16 copyright_offset` (header size is this
//! value + 4, header ends with `"(c)CRI"`), `0x04: u8 encoding`,
//! `0x05: u8 frame_size` (18), `0x06: u8 bits_per_sample` (4),
//! `0x07: u8 channels`, `0x08: u32 sample_rate`, `0x0C: u32 total_samples`,
//! `0x10: u16 highpass_frequency`, `0x12: u16 version` (`0x0300`/`0x0400`/
//! `0x0500`), `0x14: u8 encryption` (v4+: 0, 8 or 9), then per-channel
//! history `(i16 s1, i16 s2)` at `0x18`, then an optional 26-byte loop
//! extension (`pad u16, align u16, enabled u32, start_sample u32,
//! start_byte u32, end_sample u32, end_byte u32, zero u16`).
//!
//! A frame is 18 bytes: a big-endian `u16` scale followed by 32 packed
//! signed 4-bit residuals. Decoding is exact fixed point:
//! `pred = (c1*s1 + c2*s2) >> 12; sample = clamp16(nibble*scale + pred)`
//! where `c1`/`c2` are the highpass coefficients scaled by 4096.

use crate::math;
use crate::{Error, Result};

/// ADX magic at offset 0.
pub const ADX_MAGIC: u16 = 0x8000;
/// Samples per ADX frame per channel.
pub const SAMPLES_PER_FRAME: usize = 32;
/// Bytes per ADX frame per channel.
pub const FRAME_SIZE: usize = 18;
/// Maximum channels supported without allocation.
pub const MAX_CHANNELS: usize = 8;
/// ADX end-marker frame.
pub const END_MARKER: [u8; FRAME_SIZE] = [
    0x80, 0x01, 0x00, 0x0e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// ADX encoding type (byte 4 of the header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AdxEncoding {
    /// Type 2: fixed filter coefficients.
    FixedCoeff,
    /// Type 3: standard 4-bit ADX ADPCM (coefficients from cutoff).
    Standard,
    /// Type 4: exponential scale (scale field is a power-of-two exponent).
    Exponential,
    /// Type 5: AHX (MPEG-derived; recognised in headers, not decoded here).
    Ahx,
}

impl AdxEncoding {
    /// Convert a raw header byte to an encoding, if known.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            2 => Some(Self::FixedCoeff),
            3 => Some(Self::Standard),
            4 => Some(Self::Exponential),
            5 => Some(Self::Ahx),
            _ => None,
        }
    }
    /// Raw header byte for this encoding.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::FixedCoeff => 2,
            Self::Standard => 3,
            Self::Exponential => 4,
            Self::Ahx => 5,
        }
    }
}

/// ADX encryption type (byte `0x14` in v4 / v5 headers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncryptionType {
    /// No encryption.
    None,
    /// Type 8 LCG key stream.
    Type8,
    /// Type 9 LCG key stream.
    Type9,
}

impl EncryptionType {
    /// Convert a raw header byte, if known.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::None),
            8 => Some(Self::Type8),
            9 => Some(Self::Type9),
            _ => None,
        }
    }
    /// Raw header byte.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Type8 => 8,
            Self::Type9 => 9,
        }
    }
}

/// Loop points carried by an ADX header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdxLoop {
    /// First looped sample (per channel).
    pub start_sample: u32,
    /// Byte offset of the loop start in the file.
    pub start_byte: u32,
    /// Sample one past the loop end (per channel).
    pub end_sample: u32,
    /// Byte offset one past the loop end.
    pub end_byte: u32,
}

/// Parsed ADX header (owned, no borrow of the input).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdxHeader {
    /// Encoding type.
    pub encoding: AdxEncoding,
    /// Raw encoding byte.
    pub encoding_byte: u8,
    /// Frame size in bytes (18 for standard ADX).
    pub frame_size: u8,
    /// Bits per sample (4).
    pub bits_per_sample: u8,
    /// Channel count (1..=8).
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total samples per channel.
    pub total_samples: u32,
    /// Highpass cutoff frequency in Hz.
    pub highpass_frequency: u16,
    /// Version major (3, 4 or 5).
    pub version: u16,
    /// Encryption type.
    pub encryption: EncryptionType,
    /// Header size in bytes (audio data starts here).
    pub header_size: usize,
    /// Initial filter history `(s1, s2)` per channel (padded to 8).
    pub initial_history: [(i16, i16); MAX_CHANNELS],
    /// Number of valid history entries (`max(channels, 2)` in files).
    pub history_count: usize,
    /// Optional loop points.
    pub loop_points: Option<AdxLoop>,
    /// Filter coefficient 1, scaled by 4096.
    pub coeff1: i32,
    /// Filter coefficient 2, scaled by 4096.
    pub coeff2: i32,
}

/// Compute highpass filter coefficients scaled by 4096.
///
/// Uses `a = sqrt(2) - cos(2*pi*cutoff/rate)`, `b = sqrt(2) - 1`,
/// `c = (a - sqrt((a+b)*(a-b))) / b`, then `c1 = 2c`, `c2 = -c*c`.
pub fn coefficients(sample_rate: u32, highpass: u16) -> (i32, i32) {
    if sample_rate == 0 || highpass == 0 {
        return (0, 0);
    }
    let sqrt2 = math::sqrt_f32(2.0);
    let a = sqrt2 - math::cos_f32(2.0 * math::PI * highpass as f32 / sample_rate as f32);
    let b = sqrt2 - 1.0;
    let disc = (a + b) * (a - b);
    let c = if disc > 0.0 && b != 0.0 {
        (a - math::sqrt_f32(disc)) / b
    } else {
        0.0
    };
    ((2.0 * c * 4096.0) as i32, (-(c * c) * 4096.0) as i32)
}

/// Fixed coefficients used by encoding type 2.
pub const FIXED_COEFF: (i32, i32) = (1792, -768);

fn be_u16(b: &[u8], at: usize) -> Result<u16> {
    let s = b.get(at..at + 2).ok_or(Error::UnexpectedEof)?;
    Ok(u16::from_be_bytes([s[0], s[1]]))
}
fn be_u32(b: &[u8], at: usize) -> Result<u32> {
    let s = b.get(at..at + 4).ok_or(Error::UnexpectedEof)?;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

impl AdxHeader {
    /// Parse an ADX header from the start of `bytes`.
    #[allow(clippy::needless_range_loop)]
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 0x20 {
            return Err(Error::UnexpectedEof);
        }
        if be_u16(bytes, 0)? != ADX_MAGIC {
            return Err(Error::InvalidMagic);
        }
        let header_size =
            (be_u16(bytes, 2)? as usize)
                .checked_add(4)
                .ok_or(Error::InvalidField {
                    what: "header_size",
                    value: 0,
                })?;
        if header_size < 0x20 || header_size > bytes.len() {
            return Err(Error::InvalidField {
                what: "header_size",
                value: header_size as u64,
            });
        }
        let header = &bytes[..header_size];
        if !header.ends_with(b"(c)CRI") {
            return Err(Error::MissingCopyright);
        }
        let encoding_byte = header[4];
        let encoding =
            AdxEncoding::from_byte(encoding_byte).ok_or(Error::Unsupported("ADX encoding type"))?;
        let frame_size = header[5];
        let bits_per_sample = header[6];
        let channels = header[7];
        if frame_size as usize != FRAME_SIZE {
            return Err(Error::InvalidField {
                what: "frame_size",
                value: frame_size as u64,
            });
        }
        if bits_per_sample != 4 {
            return Err(Error::InvalidField {
                what: "bits_per_sample",
                value: bits_per_sample as u64,
            });
        }
        if channels == 0 || channels as usize > MAX_CHANNELS {
            return Err(Error::InvalidField {
                what: "channels",
                value: channels as u64,
            });
        }
        let sample_rate = be_u32(header, 8)?;
        let total_samples = be_u32(header, 12)?;
        if sample_rate == 0 {
            return Err(Error::InvalidField {
                what: "sample_rate",
                value: 0,
            });
        }
        let highpass_frequency = be_u16(header, 16)?;
        let version_raw = be_u16(header, 18)?;
        let version = match version_raw {
            0x0300 => 3,
            0x0400 => 4,
            0x0500 => 5,
            3..=5 => version_raw,
            other => return Err(Error::UnsupportedVersion(other as u32)),
        };
        let encryption = if version >= 4 {
            EncryptionType::from_byte(header[20])
                .ok_or(Error::Unsupported("ADX encryption type"))?
        } else {
            EncryptionType::None
        };
        let history_count = (channels as usize).max(2);
        let history_end = 0x18usize
            .checked_add(history_count * 4)
            .ok_or(Error::InvalidField {
                what: "history",
                value: 0,
            })?;
        if history_end > header_size.saturating_sub(6) {
            return Err(Error::UnexpectedEof);
        }
        let mut initial_history = [(0i16, 0i16); MAX_CHANNELS];
        for i in 0..history_count.min(MAX_CHANNELS) {
            let at = 0x18 + i * 4;
            initial_history[i] = (be_u16(header, at)? as i16, be_u16(header, at + 2)? as i16);
        }
        // Loop extension sits between history and the "(c)CRI" terminator.
        let marker_at = header_size - 6;
        let remaining = marker_at.saturating_sub(history_end);
        let loop_points = match remaining {
            0 => None,
            2 => {
                if be_u16(header, history_end)? != 0 {
                    return Err(Error::InvalidField {
                        what: "loop terminator",
                        value: 0,
                    });
                }
                None
            }
            26 | 28 => {
                // Real CRI v4 loop extension is 26 bytes (0x1a):
                // padding(u16) + alignment(u16) + enabled(u32) + start_sample(u32)
                // + start_byte(u32) + end_sample(u32) + end_byte(u32) + zero(u16).
                let at = history_end;
                let _padding = be_u16(header, at)?;
                let _alignment = be_u16(header, at + 2)?;
                let enabled = be_u32(header, at + 4)?;
                if be_u16(header, at + 24)? != 0 {
                    return Err(Error::InvalidField {
                        what: "loop terminator",
                        value: 0,
                    });
                }
                if enabled == 0 {
                    None
                } else {
                    let lp = AdxLoop {
                        start_sample: be_u32(header, at + 8)?,
                        start_byte: be_u32(header, at + 12)?,
                        end_sample: be_u32(header, at + 16)?,
                        end_byte: be_u32(header, at + 20)?,
                    };
                    if lp.start_sample >= lp.end_sample || lp.end_sample > total_samples {
                        return Err(Error::InvalidField {
                            what: "loop samples",
                            value: lp.end_sample as u64,
                        });
                    }
                    Some(lp)
                }
            }
            _ => {
                return Err(Error::InvalidField {
                    what: "header extension",
                    value: remaining as u64,
                });
            }
        };
        let (coeff1, coeff2) = match encoding {
            AdxEncoding::FixedCoeff => FIXED_COEFF,
            _ => coefficients(sample_rate, highpass_frequency),
        };
        Ok(Self {
            encoding,
            encoding_byte,
            frame_size,
            bits_per_sample,
            channels,
            sample_rate,
            total_samples,
            highpass_frequency,
            version,
            encryption,
            header_size,
            initial_history,
            history_count,
            loop_points,
            coeff1,
            coeff2,
        })
    }

    /// Number of 32-sample frame groups in the stream.
    pub fn frame_groups(&self) -> usize {
        (self.total_samples as usize + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME
    }

    /// Byte length of the frame stream (excluding header / end marker).
    pub fn data_len(&self) -> usize {
        self.frame_groups() * self.channels as usize * FRAME_SIZE
    }
}

/// Zero-allocation borrowed view of a complete ADX stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdxView<'a> {
    /// Parsed header.
    pub header: AdxHeader,
    /// Interleaved frame bytes (one 18-byte frame per channel per group).
    pub frames: &'a [u8],
    /// Trailing end-marker frame, if present and valid.
    pub end_marker: Option<&'a [u8]>,
    raw: &'a [u8],
}

impl<'a> AdxView<'a> {
    /// Parse and bound a complete ADX stream.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = AdxHeader::parse(bytes)?;
        let data_len = header.data_len();
        let start = header.header_size;
        let end = start.checked_add(data_len).ok_or(Error::InvalidField {
            what: "data_len",
            value: data_len as u64,
        })?;
        let frames = bytes.get(start..end).ok_or(Error::UnexpectedEof)?;
        let end_marker = bytes.get(end..end + FRAME_SIZE);
        Ok(Self {
            header,
            frames,
            end_marker,
            raw: bytes,
        })
    }

    /// Original bytes.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.raw
    }

    /// One channel frame within a group.
    pub fn frame(&self, group: usize, channel: usize) -> Option<&'a [u8]> {
        let idx = group * self.header.channels as usize + channel;
        self.frames
            .get(idx * FRAME_SIZE..idx * FRAME_SIZE + FRAME_SIZE)
    }
}

/// 15-bit LCG key stream for CRI ADX encryption types 8 and 9.
///
/// CRI ADX type 8 and type 9 encrypt each frame's 13-bit scale header
/// (`bytes[0..2] & 0x1FFF`) with `xor & 0x1FFF`, stepping
/// `xor = (xor * mult + inc) & 0x7FFF` once per channel frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LcgKeyStream {
    state: u16,
    initial: u16,
    mult: u16,
    inc: u16,
    kind: EncryptionType,
}

impl LcgKeyStream {
    /// Create a key stream from explicit CRI LCG parameters `(start, mult, inc)`.
    pub fn from_params(kind: EncryptionType, start: u16, mult: u16, inc: u16) -> Self {
        let s = start & 0x7FFF;
        Self {
            state: s,
            initial: s,
            mult: mult & 0x7FFF,
            inc: inc & 0x7FFF,
            kind,
        }
    }

    /// Create a key stream for `kind` from a 64-bit key (deriving `(start, mult, inc)`).
    pub fn new(kind: EncryptionType, key: u64) -> Self {
        let lo = (key as u32) ^ ((key >> 32) as u32);
        // Odd multiplier (mod 4 == 1) and odd increment ensure full 15-bit period.
        let (start, mult, inc) = match kind {
            EncryptionType::Type8 => (
                ((lo as u16) ^ 0x4567) & 0x7FFF,
                ((((lo >> 16) as u16) | 1) & 0x7FFD) | 1,
                (((lo >> 8) as u16) | 1) & 0x7FFF,
            ),
            EncryptionType::Type9 => (
                ((lo as u16) ^ 0x1357) & 0x7FFF,
                (((((lo >> 16) as u16) ^ 0x0D19) & 0x7FFC) | 1),
                ((((lo >> 8) as u16) ^ 0x043B) | 1) & 0x7FFF,
            ),
            EncryptionType::None => (0, 1, 0),
        };
        Self::from_params(kind, start, mult, inc)
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.state = self.initial;
    }

    /// Return the current 13-bit scale XOR mask and advance the 15-bit LCG by one frame.
    pub fn next_scale_mask(&mut self) -> u16 {
        if self.kind == EncryptionType::None {
            return 0;
        }
        let mask = self.state & 0x1FFF;
        self.state = ((self.state as u32)
            .wrapping_mul(self.mult as u32)
            .wrapping_add(self.inc as u32)
            & 0x7FFF) as u16;
        mask
    }

    /// Return the low 8 bits of the next 15-bit LCG step (compatibility helper).
    pub fn next_byte(&mut self) -> u8 {
        self.next_scale_mask() as u8
    }
}

/// XOR one ADX channel frame's 13-bit scale field (bytes `0..2`) with the LCG key stream.
pub fn xor_frame(frame: &mut [u8], stream: &mut LcgKeyStream) {
    if frame.len() >= 2 && stream.kind != EncryptionType::None {
        let raw = u16::from_be_bytes([frame[0], frame[1]]);
        let mask = stream.next_scale_mask();
        let xored = (raw & 0xE000) | ((raw ^ mask) & 0x1FFF);
        frame[0..2].copy_from_slice(&xored.to_be_bytes());
    }
}

fn clamp16(v: i32) -> i16 {
    v.clamp(-32768, 32767) as i16
}

fn signed_nibble(n: u8) -> i32 {
    let n = (n & 0x0F) as i32;
    if n >= 8 {
        n - 16
    } else {
        n
    }
}

/// Decode one 18-byte channel frame into 32 samples, updating history.
///
/// No allocation. Returns `Err(Unsupported)` for AHX frames.
pub fn decode_frame(
    frame: &[u8],
    header: &AdxHeader,
    hist: &mut (i32, i32),
    out: &mut [i16; SAMPLES_PER_FRAME],
) -> Result<()> {
    if frame.len() < FRAME_SIZE {
        return Err(Error::UnexpectedEof);
    }
    if header.encoding == AdxEncoding::Ahx {
        return Err(Error::Unsupported("AHX decoding"));
    }
    let scale_raw = (u16::from_be_bytes([frame[0], frame[1]]) & 0x1FFF) as i32;
    let scale = scale_raw + 1;
    for i in 0..SAMPLES_PER_FRAME {
        let byte = frame[2 + i / 2];
        let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
        let residual = match header.encoding {
            AdxEncoding::Exponential => {
                let exp = (scale_raw & 0x0F).min(15);
                signed_nibble(nib) * (1i32 << exp)
            }
            _ => signed_nibble(nib) * scale,
        };
        let pred = (header.coeff1 * hist.0 + header.coeff2 * hist.1) >> 12;
        let sample = clamp16(residual + pred);
        out[i] = sample;
        hist.1 = hist.0;
        hist.0 = sample as i32;
    }
    Ok(())
}

/// Streaming ADX decoder over a borrowed stream.
#[derive(Debug, Clone)]
pub struct AdxDecoder<'a> {
    view: AdxView<'a>,
    hist: [(i32, i32); MAX_CHANNELS],
    group: usize,
    key_stream: Option<LcgKeyStream>,
    key: u64,
    key_params: Option<(u16, u16, u16)>,
    looping: bool,
    pending: [i16; SAMPLES_PER_FRAME * MAX_CHANNELS],
    pending_frames: usize,
    pending_pos: usize,
    finished: bool,
}

impl<'a> AdxDecoder<'a> {
    /// Create a decoder from a complete ADX stream.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let view = AdxView::parse(bytes)?;
        let mut dec = Self {
            hist: [(0, 0); MAX_CHANNELS],
            group: 0,
            key_stream: None,
            key: 0,
            key_params: None,
            looping: view.header.loop_points.is_some(),
            pending: [0; SAMPLES_PER_FRAME * MAX_CHANNELS],
            pending_frames: 0,
            pending_pos: 0,
            finished: false,
            view,
        };
        dec.reset()?;
        Ok(dec)
    }

    /// Parsed header.
    pub fn header(&self) -> &AdxHeader {
        &self.view.header
    }

    /// Channel count.
    pub fn channels(&self) -> usize {
        self.view.header.channels as usize
    }

    /// Sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.view.header.sample_rate
    }

    /// Enable or disable seamless looping (default: on if loop points exist).
    pub fn set_looping(&mut self, on: bool) {
        self.looping = on;
    }

    /// Supply the decryption key for type 8 / 9 streams and reset.
    pub fn set_key(&mut self, key: u64) -> Result<()> {
        self.key = key;
        self.key_params = None;
        self.reset()
    }

    /// Supply explicit CRI type 8 / 9 LCG parameters `(start, mult, inc)` and reset.
    pub fn set_key_params(&mut self, start: u16, mult: u16, inc: u16) -> Result<()> {
        self.key_params = Some((start, mult, inc));
        self.reset()
    }

    /// Reset to the start, restoring initial history and the key stream.
    pub fn reset(&mut self) -> Result<()> {
        for i in 0..MAX_CHANNELS {
            let (s1, s2) = self.view.header.initial_history[i];
            self.hist[i] = (s1 as i32, s2 as i32);
        }
        self.group = 0;
        self.pending_frames = 0;
        self.pending_pos = 0;
        self.finished = false;
        self.key_stream = match self.view.header.encryption {
            EncryptionType::None => None,
            kind => Some(match self.key_params {
                Some((start, mult, inc)) => LcgKeyStream::from_params(kind, start, mult, inc),
                None => LcgKeyStream::new(kind, self.key),
            }),
        };
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn decode_next_group(&mut self) -> Result<bool> {
        let total_groups = self.view.header.frame_groups();
        if self.group >= total_groups {
            if self.looping {
                if let Some(lp) = self.view.header.loop_points {
                    self.group = (lp.start_sample as usize) / SAMPLES_PER_FRAME;
                    // Reposition key stream to the loop-start group (1 step per channel frame).
                    let skip = self.group * self.channels();
                    if let Some(ks) = &mut self.key_stream {
                        ks.reset();
                        for _ in 0..skip {
                            ks.next_scale_mask();
                        }
                    }
                } else {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        if self.view.header.encoding == AdxEncoding::Ahx {
            return Err(Error::Unsupported("AHX decoding"));
        }
        let ch = self.channels();
        let mut tmp = [0i16; SAMPLES_PER_FRAME];
        for c in 0..ch {
            let frame = self.view.frame(self.group, c).ok_or(Error::UnexpectedEof)?;
            let mut owned = [0u8; FRAME_SIZE];
            owned.copy_from_slice(frame);
            if let Some(ks) = &mut self.key_stream {
                xor_frame(&mut owned, ks);
            }
            let mut h = self.hist[c];
            decode_frame(&owned, &self.view.header, &mut h, &mut tmp)?;
            self.hist[c] = h;
            for i in 0..SAMPLES_PER_FRAME {
                self.pending[i * ch + c] = tmp[i];
            }
        }
        // Trim the final group to total_samples.
        let remaining_total =
            self.view.header.total_samples as usize - self.group * SAMPLES_PER_FRAME;
        self.pending_frames = remaining_total.min(SAMPLES_PER_FRAME);
        self.pending_pos = 0;
        self.group += 1;
        Ok(true)
    }

    /// Decode interleaved `i16` PCM. Returns sample-frames written per channel.
    #[allow(clippy::collapsible_if)]
    pub fn decode_i16(&mut self, out: &mut [i16]) -> Result<usize> {
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
                if !self.decode_next_group()? {
                    self.finished = true;
                    break;
                }
            }
            let avail = self.pending_frames - self.pending_pos;
            let take = avail.min(capacity - written);
            for i in 0..take {
                let src = (self.pending_pos + i) * ch;
                out[(written + i) * ch..(written + i) * ch + ch]
                    .copy_from_slice(&self.pending[src..src + ch]);
            }
            self.pending_pos += take;
            written += take;
        }
        Ok(written)
    }

    /// Decode planar `f32` PCM (`planes[c]` per channel). Returns frames written.
    pub fn decode_f32_planar(&mut self, planes: &mut [&mut [f32]]) -> Result<usize> {
        let ch = self.channels();
        if planes.len() < ch {
            return Err(Error::BufferTooSmall {
                needed: ch,
                actual: planes.len(),
            });
        }
        let capacity = planes.iter().take(ch).map(|p| p.len()).min().unwrap_or(0);
        // Decode via a fixed interleaved scratch on the stack (no allocation).
        let mut scratch = [0i16; 256 * MAX_CHANNELS];
        let want = capacity.min(256);
        if want == 0 {
            return Ok(0);
        }
        let got = self.decode_i16(&mut scratch[..want * ch])?;
        for c in 0..ch {
            for i in 0..got {
                planes[c][i] = scratch[i * ch + c] as f32 / 32768.0;
            }
        }
        Ok(got)
    }

    /// Decode interleaved `f32` PCM. Returns frames written.
    pub fn decode_f32(&mut self, out: &mut [f32]) -> Result<usize> {
        let ch = self.channels();
        let capacity = out.len().checked_div(ch).unwrap_or(0);
        let mut scratch = [0i16; 256 * MAX_CHANNELS];
        let want = capacity.min(256);
        if want == 0 {
            return Ok(0);
        }
        let got = self.decode_i16(&mut scratch[..want * ch])?;
        for i in 0..got * ch {
            out[i] = scratch[i] as f32 / 32768.0;
        }
        Ok(got)
    }
}

/// Stateful ADX encoder (closed-loop, mirrors the decoder's prediction).
#[derive(Debug, Clone)]
pub struct AdxEncoder {
    header_template: AdxHeader,
    hist: [(i32, i32); MAX_CHANNELS],
    key_stream: Option<LcgKeyStream>,
}

impl AdxEncoder {
    /// Create an encoder. `total_samples` in the template is informational.
    pub fn new(
        channels: u8,
        sample_rate: u32,
        highpass: u16,
        encoding: AdxEncoding,
        encryption: EncryptionType,
        key: u64,
    ) -> Result<Self> {
        if channels == 0 || channels as usize > MAX_CHANNELS || sample_rate == 0 {
            return Err(Error::InvalidParameter("channels/sample_rate"));
        }
        let (coeff1, coeff2) = match encoding {
            AdxEncoding::FixedCoeff => FIXED_COEFF,
            _ => coefficients(sample_rate, highpass),
        };
        Ok(Self {
            header_template: AdxHeader {
                encoding,
                encoding_byte: encoding.to_byte(),
                frame_size: FRAME_SIZE as u8,
                bits_per_sample: 4,
                channels,
                sample_rate,
                total_samples: 0,
                highpass_frequency: highpass,
                version: 4,
                encryption,
                header_size: 0,
                initial_history: [(0, 0); MAX_CHANNELS],
                history_count: (channels as usize).max(2),
                loop_points: None,
                coeff1,
                coeff2,
            },
            hist: [(0, 0); MAX_CHANNELS],
            key_stream: match encryption {
                EncryptionType::None => None,
                kind => Some(LcgKeyStream::new(kind, key)),
            },
        })
    }

    /// Reset encoder history and key stream.
    pub fn reset(&mut self, key: u64) {
        self.hist = [(0, 0); MAX_CHANNELS];
        self.key_stream = match self.header_template.encryption {
            EncryptionType::None => None,
            kind => Some(LcgKeyStream::new(kind, key)),
        };
    }

    /// Header template used for coefficient / encoding decisions.
    pub fn header_template(&self) -> &AdxHeader {
        &self.header_template
    }

    /// Encode one group: `input` holds `32 * channels` interleaved samples,
    /// `out` receives `18 * channels` bytes. Returns bytes written.
    pub fn encode_group(&mut self, input: &[i16], out: &mut [u8]) -> Result<usize> {
        let ch = self.header_template.channels as usize;
        if input.len() < SAMPLES_PER_FRAME * ch {
            return Err(Error::BufferTooSmall {
                needed: SAMPLES_PER_FRAME * ch,
                actual: input.len(),
            });
        }
        if out.len() < FRAME_SIZE * ch {
            return Err(Error::BufferTooSmall {
                needed: FRAME_SIZE * ch,
                actual: out.len(),
            });
        }
        if self.header_template.encoding == AdxEncoding::Ahx {
            return Err(Error::Unsupported("AHX encoding"));
        }
        for c in 0..ch {
            let mut samples = [0i16; SAMPLES_PER_FRAME];
            for i in 0..SAMPLES_PER_FRAME {
                samples[i] = input[i * ch + c];
            }
            let frame = &mut out[c * FRAME_SIZE..c * FRAME_SIZE + FRAME_SIZE];
            encode_channel_frame(&samples, &self.header_template, &mut self.hist[c], frame);
            if let Some(ks) = &mut self.key_stream {
                xor_frame(frame, ks);
            }
        }
        Ok(FRAME_SIZE * ch)
    }
}

fn encode_channel_frame(
    samples: &[i16; SAMPLES_PER_FRAME],
    header: &AdxHeader,
    hist: &mut (i32, i32),
    frame: &mut [u8],
) {
    // First pass: residuals against a simulated (quantised) history.
    let mut sim = *hist;
    let mut residuals = [0i32; SAMPLES_PER_FRAME];
    let mut max_abs = 0i32;
    for i in 0..SAMPLES_PER_FRAME {
        let pred = (header.coeff1 * sim.0 + header.coeff2 * sim.1) >> 12;
        let r = samples[i] as i32 - pred;
        residuals[i] = r;
        max_abs = max_abs.max(r.abs());
        // Placeholder advance; refined in the quantisation pass below.
        sim.1 = sim.0;
        sim.0 = samples[i] as i32;
    }
    let (scale_field, denom): (u16, i32) = match header.encoding {
        AdxEncoding::Exponential => {
            let mut e = 0i32;
            while (7 << e) < max_abs && e < 15 {
                e += 1;
            }
            (e as u16, 1 << e)
        }
        _ => {
            let s = if max_abs == 0 {
                1
            } else {
                ((max_abs + 6) / 7).clamp(1, 0x2000)
            };
            ((s - 1) as u16, s)
        }
    };
    frame[0..2].copy_from_slice(&scale_field.to_be_bytes());
    // Second pass: quantise with closed-loop history so decoder matches.
    let mut nibbles = [0u8; SAMPLES_PER_FRAME];
    for i in 0..SAMPLES_PER_FRAME {
        let pred = (header.coeff1 * hist.0 + header.coeff2 * hist.1) >> 12;
        let r = samples[i] as i32 - pred;
        let q = if denom == 0 {
            0
        } else if header.encoding == AdxEncoding::Exponential {
            (r.div_euclid(denom)).clamp(-8, 7)
        } else {
            ((r + if r >= 0 { denom / 2 } else { -denom / 2 }) / denom).clamp(-8, 7)
        };
        nibbles[i] = (q & 0x0F) as u8;
        let reconstructed = clamp16(q * denom + pred);
        hist.1 = hist.0;
        hist.0 = reconstructed as i32;
        let _ = residuals[i];
    }
    for i in 0..16 {
        frame[2 + i] = (nibbles[2 * i] << 4) | nibbles[2 * i + 1];
    }
}

/// Build a complete synthetic ADX file header (with optional loop points).
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
pub fn build_header(
    channels: u8,
    sample_rate: u32,
    total_samples: u32,
    highpass: u16,
    version: u16,
    encoding: AdxEncoding,
    encryption: EncryptionType,
    loop_points: Option<AdxLoop>,
) -> Result<alloc::vec::Vec<u8>> {
    let history_count = (channels as usize).max(2);
    let ext_len = if loop_points.is_some() { 26 } else { 2 };
    let header_size = 0x18 + history_count * 4 + ext_len + 6;
    let mut h = alloc::vec![0u8; header_size];
    h[0..2].copy_from_slice(&ADX_MAGIC.to_be_bytes());
    h[2..4].copy_from_slice(&((header_size - 4) as u16).to_be_bytes());
    h[4] = encoding.to_byte();
    h[5] = FRAME_SIZE as u8;
    h[6] = 4;
    h[7] = channels;
    h[8..12].copy_from_slice(&sample_rate.to_be_bytes());
    h[12..16].copy_from_slice(&total_samples.to_be_bytes());
    h[16..18].copy_from_slice(&highpass.to_be_bytes());
    let vraw = match version {
        3 => 0x0300u16,
        4 => 0x0400,
        5 => 0x0500,
        other => return Err(Error::UnsupportedVersion(other as u32)),
    };
    h[18..20].copy_from_slice(&vraw.to_be_bytes());
    if version >= 4 {
        h[20] = encryption.to_byte();
    }
    let at = 0x18 + history_count * 4;
    if let Some(lp) = loop_points {
        h[at + 2..at + 4].copy_from_slice(&1u16.to_be_bytes());
        h[at + 4..at + 8].copy_from_slice(&1u32.to_be_bytes());
        h[at + 8..at + 12].copy_from_slice(&lp.start_sample.to_be_bytes());
        h[at + 12..at + 16].copy_from_slice(&lp.start_byte.to_be_bytes());
        h[at + 16..at + 20].copy_from_slice(&lp.end_sample.to_be_bytes());
        h[at + 20..at + 24].copy_from_slice(&lp.end_byte.to_be_bytes());
    }
    let n = h.len();
    h[n - 6..].copy_from_slice(b"(c)CRI");
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_zero_cutoff() {
        assert_eq!(coefficients(44100, 0), (0, 0));
        let (c1, c2) = coefficients(44100, 500);
        assert!(c1 > 0 && c2 < 0, "{c1} {c2}");
    }

    #[test]
    fn lcg_deterministic_and_differs_by_type() {
        let mut a = LcgKeyStream::new(EncryptionType::Type8, 42);
        let mut b = LcgKeyStream::new(EncryptionType::Type8, 42);
        let mut c = LcgKeyStream::new(EncryptionType::Type9, 42);
        for _ in 0..16 {
            assert_eq!(a.next_byte(), b.next_byte());
        }
        let mut a2 = LcgKeyStream::new(EncryptionType::Type8, 42);
        assert_ne!(a2.next_byte(), c.next_byte());
    }
}
