use std::path::Path;
use larql_models::loading::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let gguf = GgufFile::open(path)?;
    
    println!("Data Offset: {}", gguf.data_offset);
    
    for info in &gguf.tensor_infos {
        if info.name == "token_embd.weight" {
            println!("TENSOR: {}: dims={:?}, type={}, offset={}", info.name, info.dims, info.tensor_type, info.offset);
            let abs_offset = gguf.data_offset + info.offset;
            println!("Abs Offset: {}", abs_offset);
            
            let mut file = std::fs::File::open(path)?;
            use std::io::{Seek, SeekFrom, Read};
            file.seek(SeekFrom::Start(abs_offset))?;
            let mut buf = [0u8; 64];
            file.read_exact(&mut buf)?;
            println!("Raw bytes: {:02X?}", buf);
        }
    }
    
    Ok(())
}
