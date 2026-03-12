import numpy as np

def load_spike_dat_binary(file_path, width=400, height=250, num_frames=20000):
    print(f"读取文件：{file_path}")

    total_bits = width * height * num_frames          # 总脉冲像素数
    total_bytes = total_bits // 8                     # 每个字节包含8个bit
    print(f"预期总字节数: {total_bytes} bytes")

    with open(file_path, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) != total_bytes:
        raise ValueError(f"文件大小与预期不�? 实际 {len(raw_data)}, 预期 {total_bytes}")

    print(f"成功读取原始数据，共 {len(raw_data)} 字节")

    # 转换�?numpy array，并展开�?bit
    byte_array = np.frombuffer(raw_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)

    if bit_array.size != total_bits:
        raise ValueError(f"位展开后尺寸不匹配，当前为 {bit_array.size}，期�?{total_bits}")

    spike_tensor = bit_array.reshape((num_frames, height, width)).astype(np.uint8)
    print(f"成功解码为张量，形状: {spike_tensor.shape}，类�? {spike_tensor.dtype}")

    return spike_tensor

# 用法示例
if __name__ == '__main__':
    dat_path = "/root/shared-nvme/MDE-SpikingCamera/Outdoor-Spike/seq_01.dat"
    spikes = load_spike_dat_binary(dat_path)
