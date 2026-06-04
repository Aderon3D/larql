use std::path::Path;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

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

fn parse_gguf(path: &Path) -> Result<(Vec<TensorInfo>, u64), Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(8))?; // skip magic + version
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
    for _ in 0..n_metadata {
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let key_len = u64::from_le_bytes(len_buf);
        let mut key_bytes = vec![0u8; key_len as usize];
        f.read_exact(&mut key_bytes)?;
        let key = String::from_utf8(key_bytes)?;
        let mut vt_buf = [0u8; 4];
        f.read_exact(&mut vt_buf)?;
        let vt = u32::from_le_bytes(vt_buf);
        println!("Metadata KV: key={}, type={}", key, vt);
        skip_gguf_value(&mut f, vt)?;
    }
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let name_len = u64::from_le_bytes(len_buf);
        let mut name_bytes = vec![0u8; name_len as usize];
        f.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)?;
        let mut n_dims_buf = [0u8; 4];
        f.read_exact(&mut n_dims_buf)?;
        let n_dims = u32::from_le_bytes(n_dims_buf);
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            let mut dim_buf = [0u8; 8];
            f.read_exact(&mut dim_buf)?;
            dims.push(u64::from_le_bytes(dim_buf));
        }
        let mut type_buf = [0u8; 4];
        f.read_exact(&mut type_buf)?;
        let tensor_type = u32::from_le_bytes(type_buf);
        let mut offset_buf = [0u8; 8];
        f.read_exact(&mut offset_buf)?;
        let offset = u64::from_le_bytes(offset_buf);
        tensors.push(TensorInfo { name, dims, tensor_type, offset });
    }
    let data_offset = f.stream_position()?;
    // align to 32 bytes or alignment value (usually 32)
    let aligned = (data_offset + 31) & !31;
    Ok((tensors, aligned))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    println!("Loading GGUF info...");
    let (tensors, data_offset) = parse_gguf(model_path)?;
    println!("Found {} tensors, data offset = {}", tensors.len(), data_offset);
    for t in &tensors {
        // Show all blk.0 tensors (full layer structure) and any global PLE tensors
        let is_blk0 = t.name.starts_with("blk.0.");
        let is_ple = t.name.contains("per_layer") || t.name.contains("token_embd");
        if is_blk0 || is_ple {
            println!("{}: dims={:?}, type={}", t.name, t.dims, t.tensor_type);
        }
    }
    Ok(())
}
