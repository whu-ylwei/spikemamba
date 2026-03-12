import numpy as np
import os
import cv2

def raw2spike(raw_seq, h, w, upside_down=False):
    """�?.dat 文件中的原始字节序列解码�?(T, H, W) 的脉冲张�?""
    raw_seq = np.array(raw_seq).astype(np.uint8)
    img_size = h * w
    img_num = len(raw_seq) // (img_size // 8)
    spk_seq = np.zeros((img_num, h, w), np.uint8)

    pix_id = np.arange(0, h * w).reshape(h, w)
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in range(img_num):
        start = img_id * (img_size // 8)
        cur_info = raw_seq[start : start + (img_size // 8)]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        spk_seq[img_id] = np.flipud(result == comparator) if upside_down else (result == comparator)

    return spk_seq.astype(np.uint8)

def save_spike_frames(dat_path, output_dir, width=400, height=250, window_size=128):
    """读取 .dat 文件，按�?window_size 帧保存一张脉冲图像（累积�?""
    print(f"读取文件：{dat_path}")
    with open(dat_path, 'rb') as f:
        raw_bytes = np.frombuffer(f.read(), dtype=np.uint8)

    spike_tensor = raw2spike(raw_bytes, height, width, upside_down=True)
    total_frames = spike_tensor.shape[0]
    print(f"脉冲张量形状: {spike_tensor.shape}，共 {total_frames} �?)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(total_frames - window_size + 1):
        window = spike_tensor[i:i + window_size]  # [128, H, W]
        fused = np.sum(window, axis=0)
        fused = np.clip(fused / window_size * 255, 0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, f"spike_{i:05d}.png")
        cv2.imwrite(out_path, fused)

        if i % 100 == 0:
            print(f"Saved frame {i}/{total_frames - window_size + 1}")

    print("�?所有脉冲图保存完毕�?)

# ==== 主函数入�?====
if __name__ == '__main__':
    dat_path = "/root/shared-nvme/MDE-SpikingCamera/Outdoor-Spike/seq_01.dat"
    output_dir = "/root/shared-nvme/MDE-SpikingCamera/output/spike_debug"
    save_spike_frames(dat_path, output_dir)
