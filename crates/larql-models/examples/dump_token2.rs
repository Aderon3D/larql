use std::path::Path;
use larql_models::loading::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let gguf = GgufFile::open(path)?;
    
    let info = gguf.tensor_infos.iter().find(|i| i.name == "token_embd.weight").unwrap();
    let abs_offset = gguf.data_offset + info.offset;
    
    let token_id = 2;
    let row_size = 1760;
    let token_offset = abs_offset + (token_id * row_size);
    
    println!("Token {} absolute offset: {}", token_id, token_offset);
    
    let mut file = std::fs::File::open(path)?;
    use std::io::{Seek, SeekFrom, Read};
    file.seek(SeekFrom::Start(token_offset))?;
    let mut buf = [0u8; 64];
    file.read_exact(&mut buf)?;
    println!("Token {} raw bytes: {:02X?}", token_id, buf);
    
    Ok(())
}
