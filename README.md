# libcri-audio

Standalone pure-Rust CRI Middleware ADX and HCA cross-console audio codec.

## Features

- **Zero runtime dependencies**, `edition = "2021"`, MSRV 1.70, `#![forbid(unsafe_code)]`, `Send + Sync`
- **ADX v3/v4/v5**: `0x8000` / `"(c)CRI"` header parsing (fixed-coeff type 2, standard type 3, exponential type 4, AHX type 5 recognition), loop metadata, type 8 / type 9 LCG encryption key streams, exact fixed-point 4-bit ADPCM decoding (18-byte frames -> 32 samples, highpass `c1`/`c2` coefficients, multi-channel, seamless looping, zero heap allocation in the hot decode loop)
- **HCA v1.3/v2.0/v3.0**: chunked header parser (`fmt`, `comp`, `dec`, `vbr`, `ath`, `loop`, `ciph`, `rva`, `comm`, `pad`) with CRC-16/BUYPASS (`0x8005`) validation, cipher types 0 / 1 / 56 (64-bit key + subkey substitution table), bit reader, ATH / intensity / scalefactor unpacking, spectral dequantization, deterministic IMDCT + overlap-add windowing to `i16` / `f32` planar PCM
- **Streaming API**: `AdxDecoder`, `HcaDecoder`, zero-allocation `AdxHeader`/`AdxView` and `HcaHeader`/`HcaView` parsers, `decode_i16`, `decode_f32_planar`, `reset()`, loop support
- `no_std` + `alloc` + `std` features (`default = ["std"]`)

All test vectors are clean-room synthetic; no copyrighted audio bytes are included.

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
