import struct

def read_string(f):
    try:
        data = f.read(8)
        if not data: return None
        length = struct.unpack('<Q', data)[0]
        return f.read(length).decode('utf-8')
    except: return None

def read_array_elem(f, etype):
    if etype == 4: return struct.unpack('<I', f.read(4))[0]
    if etype == 8: return read_string(f)
    if etype == 6: return struct.unpack('<f', f.read(4))[0]
    return "..."

def read_value(f):
    data = f.read(4)
    if not data: return None
    vtype = struct.unpack('<I', data)[0]
    if vtype == 8: # String
        return read_string(f)
    elif vtype == 4: # U32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 6: # F32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 9: # Array
        etype = struct.unpack('<I', f.read(4))[0]
        elen = struct.unpack('<Q', f.read(8))[0]
        return [read_array_elem(f, etype) for _ in range(elen)]
    else:
        # Skip other types
        sizes = {0:1, 1:1, 2:2, 3:2, 5:4, 7:1, 10:8, 11:8, 12:8}
        if vtype in sizes:
            f.read(sizes[vtype])
        return f"Type({vtype})"

path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
with open(path, "rb") as f:
    magic = f.read(4)
    if magic != b"GGUF":
        print("Not GGUF")
        exit(1)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KVs")
    for _ in range(n_kv):
        key = read_string(f)
        val = read_value(f)
        if any(x in key.lower() for x in ["head", "length", "hidden", "dim", "attn", "ffn"]):
            print(f"KV: {key}: {val}")

    for _ in range(n_tensors):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        if any(x in name.lower() for x in ["attn_q", "norm", "q_norm"]):
             print(f"Tensor: {name} {dims} {ttype}")
