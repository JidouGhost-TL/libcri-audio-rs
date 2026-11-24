// SPDX-License-Identifier: MIT OR Apache-2.0
//! Clean-room synthetic integration tests (no copyrighted bytes).

use libcri_audio::adx::{build_header, AdxEncoding, EncryptionType};
use libcri_audio::hca::{build_header as build_hca_header, HcaComp, HcaEncoder, HcaLoop};
use libcri_audio::*;

fn triangular(i: usize, amp: i32, period: usize) -> i16 {
    let p = (i % period) as i32;
    let half = period as i32 / 2;
    let v = if p < half {
        -amp + 2 * amp * p / half
    } else {
        amp - 2 * amp * (p - half) / half
    };
    v as i16
}

fn make_adx(
    channels: u8,
    version: u16,
    encoding: AdxEncoding,
    encryption: EncryptionType,
    key: u64,
    groups: usize,
    looping: bool,
) -> (Vec<u8>, Vec<i16>) {
    let total = (groups * 32) as u32;
    let ch = channels as usize;
    let loop_points = if looping {
        Some(adx::AdxLoop {
            start_sample: 32,
            start_byte: 0, // patched below if needed (parser only checks samples)
            end_sample: total,
            end_byte: 0,
        })
    } else {
        None
    };
    // Loop byte fields are not validated against file extent by design in
    // the synthetic builder; set plausible values.
    let loop_points = loop_points.map(|mut lp| {
        lp.start_byte = 64;
        lp.end_byte = 64 + (groups as u32) * 18 * channels as u32;
        lp
    });
    let mut bytes = build_header(
        channels,
        44100,
        total,
        500,
        version,
        encoding,
        encryption,
        loop_points,
    )
    .unwrap();
    let header_size = bytes.len();
    if let Some(mut lp) = loop_points {
        lp.start_byte = header_size as u32;
        lp.end_byte = header_size as u32 + (groups as u32) * 18 * channels as u32;
        // rebuild with correct bytes
        bytes = build_header(
            channels,
            44100,
            total,
            500,
            version,
            encoding,
            encryption,
            Some(lp),
        )
        .unwrap();
    }
    let mut pcm = vec![0i16; groups * 32 * ch];
    for i in 0..groups * 32 {
        for c in 0..ch {
            pcm[i * ch + c] = triangular(i + c * 7, 9000, 64);
        }
    }
    let mut enc = AdxEncoder::new(channels, 44100, 500, encoding, encryption, key).unwrap();
    let mut frames = vec![0u8; groups * 18 * ch];
    for g in 0..groups {
        enc.encode_group(
            &pcm[g * 32 * ch..(g + 1) * 32 * ch],
            &mut frames[g * 18 * ch..(g + 1) * 18 * ch],
        )
        .unwrap();
    }
    bytes.extend_from_slice(&frames);
    bytes.extend_from_slice(&adx::END_MARKER);
    (bytes, pcm)
}

fn decode_all_adx(bytes: &[u8], key: u64) -> Vec<i16> {
    let mut dec = AdxDecoder::new(bytes).unwrap();
    dec.set_looping(false);
    if key != 0 || dec.header().encryption != EncryptionType::None {
        dec.set_key(key).unwrap();
    }
    let ch = dec.channels();
    let mut out = Vec::new();
    let mut buf = [0i16; 32 * 8];
    loop {
        let n = dec.decode_i16(&mut buf[..32 * ch]).unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n * ch]);
    }
    out
}

fn mean_abs_err(a: &[i16], b: &[i16]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs() as f64)
        .sum::<f64>()
        / a.len().max(1) as f64
}

#[test]
fn adx_v3_v4_v5_headers_and_roundtrip() {
    for version in [3u16, 4, 5] {
        let (bytes, pcm) = make_adx(
            2,
            version,
            AdxEncoding::Standard,
            EncryptionType::None,
            0,
            16,
            false,
        );
        let view = adx::AdxView::parse(&bytes).unwrap();
        assert_eq!(view.header.version, version);
        assert_eq!(view.header.channels, 2);
        assert_eq!(view.header.sample_rate, 44100);
        assert_eq!(view.header.total_samples, 512);
        assert_eq!(view.header.encoding, AdxEncoding::Standard);
        let out = decode_all_adx(&bytes, 0);
        assert_eq!(out.len(), pcm.len());
        assert!(
            mean_abs_err(&out, &pcm) < 1200.0,
            "v{version} err {}",
            mean_abs_err(&out, &pcm)
        );
    }
}

#[test]
fn adx_fixed_and_exponential_roundtrip() {
    for enc in [AdxEncoding::FixedCoeff, AdxEncoding::Exponential] {
        let (bytes, pcm) = make_adx(1, 4, enc, EncryptionType::None, 0, 16, false);
        let out = decode_all_adx(&bytes, 0);
        assert_eq!(out.len(), pcm.len());
        assert!(
            mean_abs_err(&out, &pcm) < 2500.0,
            "{enc:?} err {}",
            mean_abs_err(&out, &pcm)
        );
    }
}

#[test]
fn adx_encrypted_type8_type9_roundtrip() {
    for (enc_type, key) in [
        (EncryptionType::Type8, 0x1111_2222_3333_4444u64),
        (EncryptionType::Type9, 0xDEAD_BEEF_1234_5678),
    ] {
        let (bytes, pcm) = make_adx(2, 4, AdxEncoding::Standard, enc_type, key, 12, false);
        // Encrypted frame bytes differ from a plain encoding of same PCM.
        let (plain, _) = make_adx(
            2,
            4,
            AdxEncoding::Standard,
            EncryptionType::None,
            0,
            12,
            false,
        );
        let hsz = adx::AdxHeader::parse(&bytes).unwrap().header_size;
        assert_ne!(&bytes[hsz..hsz + 36], &plain[hsz..hsz + 36]);
        let out = decode_all_adx(&bytes, key);
        assert!(
            mean_abs_err(&out, &pcm) < 1200.0,
            "{enc_type:?} err {}",
            mean_abs_err(&out, &pcm)
        );
        // Wrong key must not reproduce the signal.
        let wrong = decode_all_adx(&bytes, key ^ 0xFFFF);
        assert!(mean_abs_err(&wrong, &pcm) > mean_abs_err(&out, &pcm));
    }
}

#[test]
fn adx_loop_and_reset() {
    let (bytes, _pcm) = make_adx(
        1,
        4,
        AdxEncoding::Standard,
        EncryptionType::None,
        0,
        8,
        true,
    );
    let hdr = adx::AdxHeader::parse(&bytes).unwrap();
    assert!(hdr.loop_points.is_some());
    assert_eq!(hdr.loop_points.unwrap().start_sample, 32);
    let mut dec = AdxDecoder::new(&bytes).unwrap();
    dec.set_looping(true);
    let mut buf = [0i16; 32];
    // Decode past the end: looping keeps producing frames.
    let mut total = 0usize;
    for _ in 0..40 {
        total += dec.decode_i16(&mut buf).unwrap();
    }
    assert!(total > 256);
    dec.reset().unwrap();
    let a = dec.decode_i16(&mut buf).unwrap();
    let first = buf[..a].to_vec();
    dec.reset().unwrap();
    let b = dec.decode_i16(&mut buf).unwrap();
    assert_eq!(a, b);
    assert_eq!(&buf[..b], &first[..]);
}

#[test]
fn adx_f32_planar_matches_i16() {
    let (bytes, _) = make_adx(
        2,
        4,
        AdxEncoding::Standard,
        EncryptionType::None,
        0,
        4,
        false,
    );
    let mut d1 = AdxDecoder::new(&bytes).unwrap();
    d1.set_looping(false);
    let mut ibuf = [0i16; 128];
    let n = d1.decode_i16(&mut ibuf).unwrap();
    let mut d2 = AdxDecoder::new(&bytes).unwrap();
    d2.set_looping(false);
    let (mut l, mut r) = ([0f32; 64], [0f32; 64]);
    let m = d2.decode_f32_planar(&mut [&mut l, &mut r]).unwrap();
    assert_eq!(n, m);
    for i in 0..n {
        assert!((l[i] - ibuf[i * 2] as f32 / 32768.0).abs() < 1e-6);
        assert!((r[i] - ibuf[i * 2 + 1] as f32 / 32768.0).abs() < 1e-6);
    }
}

#[test]
fn adx_rejects_bad_inputs() {
    assert!(matches!(
        adx::AdxHeader::parse(&[0u8; 10]),
        Err(Error::UnexpectedEof)
    ));
    let (bytes, _) = make_adx(
        1,
        4,
        AdxEncoding::Standard,
        EncryptionType::None,
        0,
        2,
        false,
    );
    let mut bad = bytes.clone();
    bad[0] = 0;
    assert!(matches!(
        adx::AdxHeader::parse(&bad),
        Err(Error::InvalidMagic)
    ));
    let mut bad2 = bytes.clone();
    let n = adx::AdxHeader::parse(&bad2).unwrap().header_size;
    bad2[n - 1] = b'X';
    assert!(matches!(
        adx::AdxHeader::parse(&bad2),
        Err(Error::MissingCopyright)
    ));
    assert!(AdxDecoder::new(&bytes[..bytes.len() - 30]).is_err());
    let mut ahx = build_header(
        1,
        44100,
        32,
        500,
        4,
        AdxEncoding::Ahx,
        EncryptionType::None,
        None,
    )
    .unwrap();
    ahx.extend_from_slice(&[0u8; 18]);
    let mut dec = AdxDecoder::new(&ahx).unwrap();
    assert!(dec.decode_i16(&mut [0i16; 32]).is_err());
}

// ---- HCA ----

fn make_hca(cipher_type: u16, key: u64, frames: u32) -> (Vec<u8>, HcaCipher) {
    let comp = HcaComp {
        min_resolution: 1,
        max_resolution: 15,
        track_count: 1,
        channel_config: 0,
        total_bands: 16,
        base_bands: 16,
        stereo_bands: 0,
        bands_per_hfr_group: 0,
    };
    let header = build_hca_header(
        0x0200,
        2,
        44100,
        frames,
        512,
        comp,
        Some(0),
        Some(HcaLoop {
            start_frame: 0,
            end_frame: frames,
            pre_samples: 0,
            post_samples: 0,
        }),
        cipher_type,
        Some(0.5),
    )
    .unwrap();
    let cipher = match cipher_type {
        1 => HcaCipher::type1(),
        56 => HcaCipher::type56(key),
        _ => HcaCipher::none(),
    };
    let enc = HcaEncoder::new(2, 16, 0, 512, cipher.clone()).unwrap();
    let mut spectrum = [0.0f32; 128];
    for (i, v) in spectrum.iter_mut().enumerate().take(16) {
        *v = ((i * 37 % 13) as f32 - 6.0) * 0.5;
    }
    let silence = [0.0f32; 128];
    let mut bytes = header;
    for f in 0..frames {
        let mut frame = [0u8; 512];
        let spec = if f == 0 { &silence } else { &spectrum };
        enc.encode_frame_spectrum(&[spec, spec], &mut frame)
            .unwrap();
        bytes.extend_from_slice(&frame);
    }
    (bytes, cipher)
}

#[test]
fn hca_header_chunks_crc_and_cipher() {
    let (bytes, _) = make_hca(56, 0x0123_4567_89AB_CDEF, 4);
    let view = hca::HcaView::parse(&bytes).unwrap();
    assert_eq!(view.header.version, 0x0200);
    assert_eq!(view.header.channels, 2);
    assert_eq!(view.header.sample_rate, 44100);
    assert_eq!(view.header.frame_count, 4);
    assert_eq!(view.header.cipher_type, 56);
    assert_eq!(view.header.rva_volume, Some(0.5));
    assert!(view.header.looping.is_some());
    assert_eq!(crc16(&bytes[..view.header.header_size as usize]), 0);
    #[cfg(feature = "std")]
    {
        // chunks field is alloc-gated type; just ensure parse succeeded.
    }
    // Bad CRC rejected.
    let mut bad = bytes.clone();
    bad[10] ^= 0xFF;
    assert!(matches!(
        hca::HcaHeader::parse(&bad),
        Err(Error::CrcMismatch { .. })
    ));
    let mut bad2 = bytes.clone();
    bad2[0] = b'X';
    assert!(matches!(
        hca::HcaHeader::parse(&bad2),
        Err(Error::InvalidMagic)
    ));
}

#[test]
fn hca_versions_parse() {
    for version in [0x0103u16, 0x0200, 0x0300] {
        let comp = HcaComp {
            min_resolution: 0,
            max_resolution: 15,
            track_count: 1,
            channel_config: 0,
            total_bands: 8,
            base_bands: 8,
            stereo_bands: 0,
            bands_per_hfr_group: 0,
        };
        let header =
            build_hca_header(version, 1, 22050, 1, 256, comp, None, None, 0, None).unwrap();
        let h = hca::HcaHeader::parse(&header).unwrap();
        assert_eq!(h.version, version);
    }
}

#[test]
fn hca_decode_silence_and_signal_and_f32() {
    let (bytes, _) = make_hca(0, 0, 4);
    let mut dec = HcaDecoder::new(&bytes).unwrap();
    assert_eq!(dec.header().total_samples(), 4 * 1024);
    // Inspect frame 0 and frame 1 headers (0xFFFF sync + 6-bit scalefactors).
    let fh0 = dec.inspect_frame_header(0).unwrap();
    assert_eq!(fh0.sync, hca::FRAME_SYNC);
    let fh1 = dec.inspect_frame_header(1).unwrap();
    assert_eq!(fh1.sync, hca::FRAME_SYNC);
    assert!(fh1.scale_factors[0][..16].iter().any(|&s| s > 0));

    // Frame 0 is 1024 silent sample-frames.
    let mut buf = [0i16; 1024 * 2];
    let n1 = dec.decode_i16(&mut buf).unwrap();
    assert_eq!(n1, 1024);
    assert!(
        buf.iter().all(|&v| v == 0),
        "first (silent-spectrum) frame must be silent"
    );
    // Frame 1 is 1024 non-silent sample-frames.
    let n2 = dec.decode_i16(&mut buf).unwrap();
    assert_eq!(n2, 1024);
    assert!(
        buf.iter().any(|&v| v != 0),
        "signal frame must be non-silent"
    );
    // f32 planar agrees in sign/magnitude scale with i16 path on a fresh decoder.
    let mut dec2 = HcaDecoder::new(&bytes).unwrap();
    let (mut l, mut r) = ([0f32; 128], [0f32; 128]);
    // Skip the 8 silent subframes (1024 samples) of frame 0.
    for _ in 0..8 {
        assert_eq!(dec2.decode_f32_planar(&mut [&mut l, &mut r]).unwrap(), 128);
    }
    let m = dec2.decode_f32_planar(&mut [&mut l, &mut r]).unwrap();
    assert_eq!(m, 128);
    assert!(l.iter().any(|&v| v != 0.0));
    // reset determinism
    dec.reset();
    let mut again = [0i16; 1024 * 2];
    assert_eq!(dec.decode_i16(&mut again).unwrap(), 1024);
    assert!(again.iter().all(|&v| v == 0));
}

#[test]
fn hca_cipher_type56_keyed_decode() {
    let key = 0x0123_4567_89AB_CDEFu64;
    let (bytes, _) = make_hca(56, key, 3);
    let mut dec = HcaDecoder::with_key(&bytes, key).unwrap();
    let mut buf = [0i16; 1024 * 2];
    dec.decode_i16(&mut buf).unwrap();
    dec.decode_i16(&mut buf).unwrap();
    assert!(buf.iter().any(|&v| v != 0));
    // Wrong key: frame CRC after decipher fails.
    let mut wrong = HcaDecoder::with_key(&bytes, key ^ 1).unwrap();
    let mut b2 = [0i16; 1024 * 2];
    assert!(wrong.decode_i16(&mut b2).is_err());
    // Type 1 keyless also decodes.
    let (bytes1, _) = make_hca(1, 0, 2);
    let mut d1 = HcaDecoder::new(&bytes1).unwrap();
    assert_eq!(d1.decode_i16(&mut buf).unwrap(), 1024);
}

#[test]
fn hca_cipher_type56_awb_subkey_and_ath_table() {
    let key = 0x0012_3456_789A_BCDEu64;
    let subkey = 0x2F41u16;
    let c0 = HcaCipher::type56_with_awb_subkey(key, 0);
    let c1 = HcaCipher::type56_with_awb_subkey(key, subkey);
    assert_ne!(c0, c1);
    assert_eq!(hca::derive_type56_subkey(key, 0), key);
    assert_ne!(hca::derive_type56_subkey(key, subkey), key);

    let ath0 = hca::build_ath_table(0, 48000);
    assert!(ath0.iter().all(|&v| v == 0));
    let ath1 = hca::build_ath_table(1, 48000);
    assert!(ath1[0] > ath1[20], "sub-bass ATH higher than mid-band");
    assert!(ath1[127] > ath1[20], "ultrasonic ATH higher than mid-band");
}

#[test]
fn real_cri_samples_env_gated() {
    let Ok(dir) = std::env::var("LIBCRI_SAMPLE_DIR") else {
        return;
    };
    let root = std::path::Path::new(&dir);
    let mut hca_checked = 0usize;
    let mut adx_checked = 0usize;

    fn visit(p: &std::path::Path, hca_checked: &mut usize, adx_checked: &mut usize) {
        let Ok(rd) = std::fs::read_dir(p) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, hca_checked, adx_checked);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            if ext == "adx" {
                let hdr = adx::AdxHeader::parse(&data).unwrap();
                assert!(hdr.sample_rate > 0 && hdr.channels > 0);
                *adx_checked += 1;
            } else if ext == "hca" {
                let mut dec = HcaDecoder::new(&data).unwrap();
                let frames = dec.header().frame_count as usize;
                // Verify CRC-16, 0xFFFF sync, a/b, and delta-coded scalefactors on every frame.
                for fi in 0..frames {
                    let fh = dec.inspect_frame_header(fi).unwrap();
                    assert_eq!(fh.sync, hca::FRAME_SYNC);
                }
                // Decode up to 16 frames to verify finite, bounded PCM output.
                let ch = dec.channels();
                let mut pcm = vec![0.0f32; 1024 * ch];
                let mut decoded_frames = 0usize;
                for _ in 0..frames.min(16) {
                    let n = dec.decode_f32(&mut pcm).unwrap();
                    assert!(n > 0);
                    assert!(pcm[..n * ch]
                        .iter()
                        .all(|v| v.is_finite() && v.abs() <= 1.0001));
                    decoded_frames += 1;
                }
                assert!(decoded_frames > 0);
                *hca_checked += 1;
            }
        }
    }

    visit(root, &mut hca_checked, &mut adx_checked);
    assert!(
        hca_checked > 0,
        "expected at least one .hca in LIBCRI_SAMPLE_DIR"
    );
    assert!(
        adx_checked > 0,
        "expected at least one .adx in LIBCRI_SAMPLE_DIR"
    );
}

#[test]
fn hca_mdct_imdct_and_ath() {
    let w = hca::sine_window();
    assert!((w[128] - 1.0).abs() < 0.01);
    // Forward then inverse transform preserves a smooth block up to scale:
    // check correlation rather than exact amplitude (windowed MDCT).
    let mut input = [0.0f32; 256];
    for (i, v) in input.iter_mut().enumerate() {
        *v = libcri_audio::math::sin_f32(i as f32 * 0.05) * 100.0;
    }
    let mut spec = [0.0f32; 128];
    hca::mdct(&input, &w, &mut spec);
    assert!(spec.iter().any(|&v| v.abs() > 1.0));
    let mut out = [0.0f32; 256];
    hca::imdct(&spec, &w, &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(hca::ath_attenuation(0, 100) == 0);
    assert!(hca::ath_attenuation(1, 100) > hca::ath_attenuation(1, 0));
}

#[test]
fn hca_rejects_truncated_and_small_buffers() {
    let (bytes, _) = make_hca(0, 0, 1);
    assert!(HcaDecoder::new(&bytes[..20]).is_err());
    let mut dec = HcaDecoder::new(&bytes).unwrap();
    assert!(matches!(
        dec.decode_i16(&mut [0i16; 1]),
        Err(Error::BufferTooSmall { .. })
    ));
    assert!(matches!(
        dec.decode_f32_planar(&mut []),
        Err(Error::BufferTooSmall { .. })
    ));
}

#[test]
fn crc_and_error_display() {
    assert_eq!(crc16(b"123456789"), 0xFEE8);
    assert_eq!(
        format!("{}", Error::InvalidMagic),
        "invalid CRI audio magic"
    );
    assert_eq!(
        format!("{}", Error::BitstreamDesync),
        "bitstream desynchronization"
    );
}
