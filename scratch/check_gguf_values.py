import sys
import struct

def main():
    gguf_path = sys.argv[1]
    tensor_name = sys.argv[2]
    
    with open(gguf_path, 'rb') as f:
        # Magic
        magic = f.read(4)
        if magic != b'GGUF':
            print("Not a GGUF file")
            return
            
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Skip KV
        # This is hard without a full parser, but I just want to see if the tensor exists
        # Actually, I'll just search for the tensor name in the file and read the following bytes
        pass

if __name__ == "__main__":
    main()
