// invert.rs – Invert (or scale) selected FFN down‑projection rows in a GGUF model.

//! The tool reads a VIndex‑style patch file (JSON with a list of `Delete`
//! operations) that encodes the high‑correlation features we want to target.
//! For each `(layer, feature)` pair we locate the `ffn_down` tensor in the
//! GGUF file and multiply the scale of every Q8_0 block by the given factor.
//! By default the factor is `-1.0` (sign flip). The implementation works for
//! Q8_0 – the same format used by the Gemma‑4‑E4B model.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Patch {
    operations: Vec<PatchOp>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
enum PatchOp {
    Insert { layer: usize, feature: usize, #[serde(default)] target: String },
    Update { layer: usize, feature: usize },
    Delete { layer: usize, feature: usize },
    #[serde(other)]
    Other,
}

/// Compute the byte size of a tensor given its GGML type and element count.
fn tensor_data_size(tensor_type: u32, n_elements: usize) -> usize {
    match tensor_type {
        0 => n_elements * 4,            // F32
        1 => n_elements * 2,            // F16
        2 => n_elements / 32 * 18,      // Q4_0
        3 => n_elements / 32 * 20,      // Q4_1
        6 => n_elements / 32 * 22,      // Q5_0
        7 => n_elements / 32 * 24,      // Q5_1
        8 => n_elements / 32 * 34,      // Q8_0 – target format
        10 => n_elements / 256 * 84,    // Q2_K
        11 => n_elements / 256 * 110,   // Q3_K
        12 => n_elements / 256 * 144,   // Q4_K
        13 => n_elements / 256 * 176,   // Q5_K
        14 => n_elements / 256 * 210,   // Q6_K
        _ => panic!("Unsupported GGML type: {}", tensor_type),
    }
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

/// Advance the file cursor past a single GGUF metadata value of the given type.
/// This handles all types in the GGUF v2/v3 spec.
fn skip_gguf_value(f: &mut File, vt: u32) -> Result<(), Box<dyn std::error::Error>> {
    match vt {
        0 | 1 | 7 => { f.seek(SeekFrom::Current(1))?; }    // u8 / i8 / bool
        2 | 3     => { f.seek(SeekFrom::Current(2))?; }    // u16 / i16
        4 | 5 | 6 => { f.seek(SeekFrom::Current(4))?; }    // u32 / i32 / f32
        10 | 11 | 12 => { f.seek(SeekFrom::Current(8))?; } // u64 / i64 / f64
        8 => { // string: length(u64) + bytes
            let mut lbuf = [0u8; 8];
            f.read_exact(&mut lbuf)?;
            let slen = u64::from_le_bytes(lbuf);
            f.seek(SeekFrom::Current(slen as i64))?;
        }
        9 => { // array: element_type(u32) + count(u64) + elements
            let mut tbuf = [0u8; 4];
            f.read_exact(&mut tbuf)?;
            let elem_type = u32::from_le_bytes(tbuf);
            let mut cbuf = [0u8; 8];
            f.read_exact(&mut cbuf)?;
            let count = u64::from_le_bytes(cbuf);
            for _ in 0..count {
                skip_gguf_value(f, elem_type)?;
            }
        }
        _ => return Err(format!("Unsupported GGUF metadata value type: {}", vt).into()),
    }
    Ok(())
}

/// Minimal GGUF header parser – enough to locate tensor metadata and the start of the data section.
fn parse_gguf(path: &Path) -> Result<(Vec<TensorInfo>, u64), Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    // GGUF layout: magic(4) + version(4) + n_tensors(u64,8) + n_kv(u64,8) + kv_pairs + tensor_infos + data
    f.seek(SeekFrom::Start(8))?; // skip magic + version only
    let n_tensors = {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        u64::from_le_bytes(buf)
    };
    let n_metadata = {
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)?;
        u64::from_le_bytes(buf)
    };
    // Walk through all metadata key-value pairs, advancing the cursor without storing anything.
    for _ in 0..n_metadata {
        // key: length(u64) + bytes
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let key_len = u64::from_le_bytes(len_buf);
        f.seek(SeekFrom::Current(key_len as i64))?;
        // value type (u32)
        let mut vt_buf = [0u8; 4];
        f.read_exact(&mut vt_buf)?;
        let vt = u32::from_le_bytes(vt_buf);
        skip_gguf_value(&mut f, vt)?;
    }
    // Parse tensor descriptors
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        // name length (u64)
        let mut name_len_buf = [0u8; 8];
        f.read_exact(&mut name_len_buf)?;
        let name_len = u64::from_le_bytes(name_len_buf);
        let mut name_bytes = vec![0u8; name_len as usize];
        f.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)?;
        // n_dims (u32)
        let mut nd_buf = [0u8; 4];
        f.read_exact(&mut nd_buf)?;
        let n_dims = u32::from_le_bytes(nd_buf) as usize;
        // dims (u64 each)
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            let mut d_buf = [0u8; 8];
            f.read_exact(&mut d_buf)?;
            dims.push(u64::from_le_bytes(d_buf));
        }
        // tensor type (u32)
        let mut tt_buf = [0u8; 4];
        f.read_exact(&mut tt_buf)?;
        let tensor_type = u32::from_le_bytes(tt_buf);
        // offset (u64)
        let mut off_buf = [0u8; 8];
        f.read_exact(&mut off_buf)?;
        let offset = u64::from_le_bytes(off_buf);
        tensors.push(TensorInfo { name, dims, tensor_type, offset });
    }
    // Position now points to the beginning of raw tensor data.
    let data_offset = f.stream_position()?;
    Ok((tensors, data_offset))
}

/// Core inversion routine – applies `factor` to the scale bytes of each Q8_0 block for the selected features.
fn invert_down_weights(
    gguf_path: &Path,
    patch_path: &Path,
    factor: f32,
    inplace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load patch – we only care about Delete operations.
    let patch_json = std::fs::read_to_string(patch_path)?;
    let patch: Patch = serde_json::from_str(&patch_json)?;
    let mut deletions: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for op in patch.operations {
        if let PatchOp::Delete { layer, feature } = op {
            deletions.entry(layer).or_default().push(feature);
        }
    }
    if deletions.is_empty() {
        println!("No Delete operations found – nothing to invert.");
        return Ok(());
    }

    // Parse GGUF metadata.
    let (tensors, data_offset) = parse_gguf(gguf_path)?;

    // Decide working file.
    let work_path = if inplace {
        gguf_path.to_path_buf()
    } else {
        let mut tmp = gguf_path.to_path_buf();
        tmp.set_extension("tmp.gguf");
        std::fs::copy(gguf_path, &tmp)?;
        tmp
    };
    let mut file = OpenOptions::new().read(true).write(true).open(&work_path)?;

    for (layer, features) in deletions {
        let key = format!("blk.{}.ffn_down.weight", layer);
        let tensor = match tensors.iter().find(|t| t.name == key) {
            Some(t) => t,
            None => {
                eprintln!("Warning: tensor {} not found – skipping layer {}", key, layer);
                continue;
            }
        };
        let rows = tensor.dims.get(0).copied().unwrap_or(0) as usize;
        let cols = if tensor.dims.len() > 1 { tensor.dims[1] as usize } else { 1 };
        if rows == 0 || cols == 0 {
            eprintln!("Warning: unexpected dims for {} – rows={}, cols={}", key, rows, cols);
            continue;
        }
        let bytes_per_row = tensor_data_size(tensor.tensor_type, cols);
        let block_size = 34usize; // Q8_0: 2‑byte scale + 32‑byte payload
        let blocks_per_row = cols / 32;
        for &feat in &features {
            if feat >= rows {
                eprintln!("Feature {} out of bounds for layer {} (rows={})", feat, layer, rows);
                continue;
            }
            let row_offset = data_offset + tensor.offset + (feat as u64 * bytes_per_row as u64);
            for b in 0..blocks_per_row {
                let block_offset = row_offset + (b * block_size) as u64;
                file.seek(SeekFrom::Start(block_offset))?;
                let mut scale_bytes = [0u8; 2];
                file.read_exact(&mut scale_bytes)?;
                let new_bytes = if (factor - (-1.0)).abs() < f32::EPSILON {
                    // Sign flip – toggle sign bit (high bit of the exponent).
                    [scale_bytes[0] ^ 0x00, scale_bytes[1] ^ 0x80]
                } else {
                    let f16 = half::f16::from_bits(u16::from_le_bytes(scale_bytes));
                    let scaled = f32::from(f16) * factor;
                    let new_f16 = half::f16::from_f32(scaled);
                    new_f16.to_le_bytes()
                };
                file.seek(SeekFrom::Start(block_offset))?;
                file.write_all(&new_bytes)?;
            }
        }
    }

    if !inplace {
        std::fs::rename(&work_path, gguf_path)?;
    }
    println!("Inversion complete – factor {} applied.", factor);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: invert <model.gguf> <patch.vlp> [output.gguf] [--factor <f>] [--inplace]");
        std::process::exit(1);
    }
    let model_path = Path::new(&args[1]);
    let patch_path = Path::new(&args[2]);
    let mut inplace = false;
    let mut factor: f32 = -1.0;
    let mut output_path: Option<&Path> = None;
    let mut i = 3usize;
    while i < args.len() {
        match args[i].as_str() {
            "--inplace" => { inplace = true; i += 1; },
            "--factor" => {
                if i + 1 >= args.len() {
                    eprintln!("--factor requires a numeric argument");
                    std::process::exit(1);
                }
                factor = args[i + 1].parse::<f32>().unwrap_or(-1.0);
                i += 2;
            },
            other => {
                output_path = Some(Path::new(other));
                i += 1;
            }
        }
    }

    // If an explicit output path was given and we are not doing inplace, we will copy the result there.
    if let Some(out) = output_path {
        if !inplace {
            // Perform inversion on a temporary copy then rename to the desired output.
            invert_down_weights(model_path, patch_path, factor, false)?;
            std::fs::rename(model_path, out)?;
            return Ok(());
        }
    }

    invert_down_weights(model_path, patch_path, factor, inplace)
}
