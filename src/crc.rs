//! CRC-16/BUYPASS used by HCA headers and frames.

/// CRC-16 table for polynomial `0x8005`, MSB-first (CRC-16/BUYPASS).
pub const CRC16_TABLE: [u16; 256] = build_table();

const fn build_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute CRC-16/BUYPASS: poly `0x8005`, init `0`, no reflection, no xorout.
///
/// A well-formed HCA header (including its trailing CRC) yields `0`.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc = (crc << 8) ^ CRC16_TABLE[(((crc >> 8) as u8) ^ b) as usize];
    }
    crc
}

/// Compute the CRC value to append (big-endian) so `crc16(data ++ crc) == 0`.
pub fn crc16_append_value(data: &[u8]) -> u16 {
    crc16(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        // CRC-16/BUYPASS check value for "123456789" is 0xFEE8.
        assert_eq!(crc16(b"123456789"), 0xFEE8);
        assert_eq!(crc16(&[]), 0);
    }

    #[test]
    fn appended_crc_yields_zero() {
        let mut data = b"HCA synthetic".to_vec();
        let c = crc16(&data);
        data.extend_from_slice(&c.to_be_bytes());
        assert_eq!(crc16(&data), 0);
    }
}
