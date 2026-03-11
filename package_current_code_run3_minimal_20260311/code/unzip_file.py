import zipfile
import os

def unzip_file(zip_path, extract_to=None):
    """
    解压 .zip 文件

    参数:
    zip_path (str): .zip 文件路径
    extract_to (str): 解压目录，默认为.zip文件所在目�?    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} 不是有效�?zip 文件")

    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"已成功将 {zip_path} 解压�?{extract_to}")

# 示例用法
if __name__ == "__main__":
    zip_file_path = '/root/shared-nvme/MDE-SpikingCamera/DENSE-Spike/test/test_sequence_00_town10.zip'  # 替换为你的zip文件路径
    target_dir = '/root/shared-nvme/MDE-SpikingCamera/DENSE-Spike/test/test_sequence_00_town10'  # 或者填写路径，例如 'unzipped_folder'
    unzip_file(zip_file_path, target_dir)
