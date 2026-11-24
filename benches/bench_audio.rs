// SPDX-License-Identifier: MIT OR Apache-2.0
//! Throughput benchmark for ADX and HCA decoding (harness = false).

use std::hint::black_box;
use std::time::Instant;

use libcri_audio::adx::{build_header, AdxEncoding, EncryptionType};
use libcri_audio::hca::{build_header as build_hca_header, HcaComp, HcaEncoder};
use libcri_audio::*;

fn triangular(i: usize, amp: i32, period: usize) -> i16 {
    let p = (i % period) as i32;
    let half = period as i32 / 2;
    if p < half {
        (-amp + 2 * amp * p / half) as i16
    } else {
        (amp - 2 * amp * (p - half) / half) as i16
    }
}

fn make_adx(groups: usize) -> Vec<u8> {
    let total = (groups * 32) as u32;
    let mut bytes = build_header(
        2,
        44100,
        total,
        500,
        4,
        AdxEncoding::Standard,
        EncryptionType::None,
        None,
    )
    .unwrap();
    let mut enc = AdxEncoder::new(
        2,
        44100,
        500,
        AdxEncoding::Standard,
        EncryptionType::None,
        0,
    )
    .unwrap();
    let mut frames = vec![0u8; groups * 36];
    for g in 0..groups {
        let mut input = [0i16; 64];
        for i in 0..32 {
            input[i * 2] = triangular(g * 32 + i, 9000, 64);
            input[i * 2 + 1] = triangular(g * 32 + i + 7, 9000, 64);
        }
        enc.encode_group(&input, &mut frames[g * 36..(g + 1) * 36])
            .unwrap();
    }
    bytes.extend_from_slice(&frames);
    bytes.extend_from_slice(&adx::END_MARKER);
    bytes
}

fn make_hca(frames: u32) -> Vec<u8> {
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
    let mut bytes =
        build_hca_header(0x0200, 2, 44100, frames, 512, comp, Some(0), None, 0, None).unwrap();
    let enc = HcaEncoder::new(2, 16, 0, 512, HcaCipher::none()).unwrap();
    let mut spectrum = [0.0f32; 128];
    for (i, v) in spectrum.iter_mut().enumerate().take(16) {
        *v = ((i * 37 % 13) as f32 - 6.0) * 0.5;
    }
    for _ in 0..frames {
        let mut frame = [0u8; 512];
        enc.encode_frame_spectrum(&[&spectrum, &spectrum], &mut frame)
            .unwrap();
        bytes.extend_from_slice(&frame);
    }
    bytes
}

fn main() {
    // ADX: 10 seconds @ 44.1kHz stereo
    let groups = 44100 * 10 / 32;
    let adx_bytes = make_adx(groups);
    let mut dec = AdxDecoder::new(&adx_bytes).unwrap();
    dec.set_looping(false);
    let mut buf = [0i16; 1024 * 2];
    let t0 = Instant::now();
    let mut adx_samples: u64 = 0;
    loop {
        let n = dec.decode_i16(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        black_box(&buf);
        adx_samples += n as u64 * 2;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "bench_audio ADX: {:.2} ms, {:.3} MSamples/sec ({} samples, 2ch)",
        dt * 1000.0,
        adx_samples as f64 / dt / 1e6,
        adx_samples
    );

    // HCA: frames covering ~10 seconds (1024 samples/frame/channel)
    let frames = 44100 * 10 / 1024;
    let hca_bytes = make_hca(frames);
    let mut hdec = HcaDecoder::new(&hca_bytes).unwrap();
    let mut hbuf = [0i16; 1024 * 2];
    let t1 = Instant::now();
    let mut hca_samples: u64 = 0;
    loop {
        let n = hdec.decode_i16(&mut hbuf).unwrap();
        if n == 0 {
            break;
        }
        black_box(&hbuf);
        hca_samples += n as u64 * 2;
    }
    let dt1 = t1.elapsed().as_secs_f64();
    println!(
        "bench_audio HCA: {:.2} ms, {:.3} MSamples/sec ({} samples, 2ch)",
        dt1 * 1000.0,
        hca_samples as f64 / dt1 / 1e6,
        hca_samples
    );
}
