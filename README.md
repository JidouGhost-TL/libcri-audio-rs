# libcri-audio

Standalone pure-Rust CRI Middleware ADX and HCA cross-console audio codec.

## Features

- **Zero runtime dependencies**, `edition = "2021"`, MSRV 1.70, `#![forbid(unsafe_code)]`, `Send + Sync`
- **ADX v3/v4/v5**: `0x8000` / `"(c)CRI"` header parsing (fixed-coeff type 2, standard type 3, exponential type 4, AHX type 5 recognition), real 26-byte (`0x1a`) v4 loop extension metadata, type 8 / type 9 15-bit LCG scale-XOR encryption (`LcgKeyStream::from_params` and 64-bit key derivation), exact fixed-point 4-bit ADPCM decoding (18-byte frames -> 32 samples, `(scale & 0x1fff) + 1`, highpass `c1`/`c2` coefficients, multi-channel, seamless looping, zero heap allocation in the hot decode loop)
- **HCA v1.3/v2.0/v3.0**: chunked header parser (`fmt`, `comp`, `dec`, `vbr`, `ath`, `loop`, `ciph`, `rva`, `comm`, `pad`) with CRC-16/BUYPASS (`0x8005`) validation, 8-subframe / 1024-sample frame framing (`0xFFFF` sync, 9-bit `a`, 7-bit `b`, 3-bit `delta_bits` + 6-bit delta-coded scalefactors, 4-bit intensity, 6-bit HFR scalefactors), cipher types 0 / 1 / 56 (preserving `0x00`/`0xFF` sync anchors, with 16-bit AWB subkey derivation), and a fast O(N log N) 64-point radix-2 FFT-based 128-point IMDCT + 256-sample Princen-Bradley sine window + overlap-add to `i16` / `f32` planar PCM
- **Streaming API**: `AdxDecoder`, `HcaDecoder`, zero-allocation `AdxHeader`/`AdxView`, `HcaHeader`/`HcaView`/`HcaFrameHeader` parsers, `ImdctTables`, `decode_i16`, `decode_f32_planar`, `reset()`, loop support
- `no_std` + `alloc` + `std` features (`default = ["std"]`)

All test vectors are clean-room synthetic; no copyrighted audio bytes are included. Proprietary CRI static lookup tables (CRI's 653-entry ATH table, scale-to-resolution curve, 7 spectral Huffman codebooks, dequantizer/intensity/HFR tables, and internal Type 56 shuffle constants) are not shipped; clean-room analytical equivalents are used for spectral decoding while container parsing, CRC-16/BUYPASS, frame sync, `a`/`b`, delta-coded 6-bit scalefactors, intensity, and TDAC IMDCT follow the published framing and DSP math.

## Usage

```rust
use libcri_audio::{AdxDecoder, HcaDecoder};

let mut adx = AdxDecoder::new(&adx_bytes).unwrap();
let mut pcm = [0i16; 64];
let frames = adx.decode_i16(&mut pcm).unwrap();

let mut hca = HcaDecoder::with_key(&hca_bytes, 0x0123_4567_89AB_CDEF).unwrap();
let frames = hca.decode_i16(&mut pcm).unwrap();
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
