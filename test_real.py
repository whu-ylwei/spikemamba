import os
import json
import argparse
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from torch.utils.data import Dataset
import cv2
import numbers
from model.S2DepthNet import S2DepthTransformerUNetConv
from utils.data_augmentation import CenterCrop


def raw2spike(raw_seq, h, w, upside_down=False):
    raw_seq = np.array(raw_seq).astype(np.uint8)
    img_size = h * w
    img_num = len(raw_seq) // (img_size // 8)
    spk_seq = np.zeros((img_num, h, w), np.uint8)

    pix_id = np.arange(img_size).reshape(h, w)
    comparator = np.left_shift(1, np.mod(pix_id, 8)).astype(np.uint8)
    byte_id = pix_id // 8

    for img_id in range(img_num):
        id_start = img_id * (img_size // 8)
        id_end = id_start + (img_size // 8)
        cur_info = raw_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        spk_img = (result == comparator).astype(np.uint8)
        spk_seq[img_id] = np.flipud(spk_img) if upside_down else spk_img

    return spk_seq


def tfp(spk_seq, win_size=15, gamma=0.5):
    half_win = win_size // 2
    n, h, w = spk_seq.shape
    c_frames = np.zeros((n - win_size + 1, h, w), dtype=np.float64)
    for i in range(half_win, n - half_win):
        c_frames[i - half_win] = np.mean(spk_seq[i - half_win:i + half_win + 1], axis=0)
    return (c_frames ** gamma * 255).astype(np.uint8)


class RealSpikeDataset(Dataset):
    def __init__(self, spike_tensor, transform=None, window_size=128, step=1):
        self.tensor = spike_tensor.astype(np.float32)
        self.transform = transform
        self.window_size = window_size
        self.step = step
        self.num_frames = self.tensor.shape[0]

        # Only keep full-length windows
        self.valid_indices = []
        max_start = self.num_frames - self.window_size
        for start in range(0, max_start + 1, self.step):
            if start + self.window_size <= self.num_frames:
                self.valid_indices.append((start, start + self.window_size))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start, end = self.valid_indices[idx]
        spike_seq = self.tensor[start:end]
        if self.transform:
            spike_seq = self.transform(spike_seq)
        return torch.from_numpy(spike_seq).unsqueeze(0), spike_seq, start, end


def make_colormap(img, color_mapper):
    inv = np.amax(img) - img
    inv = np.nan_to_num(inv, nan=1)
    inv = inv / np.amax(inv)
    inv = np.nan_to_num(inv)
    rgba = color_mapper.to_rgba(inv)
    rgba[:, :, 0:3] = rgba[:, :, 0:3][..., ::-1]
    return rgba


def process_single_file(dat_path, model, color_mapper, output_root, transform, window_size=128, step=1):
    with open(dat_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    spike_tensor = raw2spike(raw, 250, 400, upside_down=True)

    total_len = spike_tensor.shape[0]
    dataset = RealSpikeDataset(spike_tensor, transform=transform, window_size=window_size, step=step)
    num_windows = len(dataset)

    base_name = os.path.splitext(os.path.basename(dat_path))[0]
    # print(f"[{base_name}] Total frames = {total_len}, window size = {window_size}, step = {step}")
    # print(f"[{base_name}] Valid sliding windows: {num_windows}")
    if num_windows == 0:
        print(f"[{base_name}] Skipped due to insufficient data.")
        return
    if total_len < window_size:
        print(f"[{base_name}] Not enough frames.")
    elif total_len - ((num_windows - 1) * step + window_size) > 0:
        remainder = total_len - ((num_windows - 1) * step + window_size)
        print(f"[{base_name}] Discarded last {remainder} frames (not enough for one window).")

    # Prepare directories
    save_dir = os.path.join(output_root, base_name)
    grey_dir = os.path.join(save_dir, 'grey')
    color_dir = os.path.join(save_dir, 'color_map')
    tfp_dir = os.path.join(save_dir, 'TFP_spike')
    os.makedirs(grey_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(tfp_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (input_tensor, spike_seq, start_idx, end_idx) in enumerate(dataset):
            # print(f"[{base_name}] Window {idx:03d}: start={start_idx}, end={end_idx}, length={end_idx - start_idx}")
            input_tensor = input_tensor.cuda()
            input_dict = {'image': input_tensor}
            output, _, _ = model(input_dict, None, {})

            output_np = output['image'][0][0].cpu().numpy()
            cv2.imwrite(os.path.join(grey_dir, f"frame_{idx:05d}.png"), output_np * 255.0)
            cm_img = make_colormap(output_np, color_mapper)
            cv2.imwrite(os.path.join(color_dir, f"frame_{idx:05d}.png"), cm_img * 255.0)

            tfp_result = tfp(spike_seq, win_size=15, gamma=0.5)
            tfp_mid = tfp_result[tfp_result.shape[0] // 2]
            cv2.imwrite(os.path.join(tfp_dir, f"frame_{idx:05d}.png"), tfp_mid)

            if idx % 100 == 0:
                print(f"[{base_name}] Saved frame {idx}")

    print(f"[{base_name}] Done.")


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_cfg = config['model']
    model_cfg['gpu'] = config['gpu']
    model = S2DepthTransformerUNetConv(model_cfg)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dat_files = [os.path.join(args.dat, f) for f in os.listdir(args.dat) if f.endswith('.dat')]
    if not dat_files:
        print("No .dat files found.")
        return

    dummy = np.random.rand(224, 224)
    vmax = np.percentile(dummy, 95)
    norm = mpl.colors.Normalize(vmin=dummy.min(), vmax=vmax)
    color_mapper = cm.ScalarMappable(norm=norm, cmap='magma')

    for dat_file in dat_files:
        process_single_file(dat_file, model, color_mapper, args.output_path, CenterCrop(224),
                            window_size=args.window_size, step=args.step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat', type=str, required=True, help='Directory containing .dat files')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint (.pth.tar)')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--output_path', type=str, default='runs/real', help='Output directory')
    parser.add_argument('--window_size', type=int, default=128, help='Window size for input')
    parser.add_argument('--step', type=int, default=128, help='Sliding step size')
    args = parser.parse_args()
    main(args)
