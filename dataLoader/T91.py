import h5py
import torch
from torch.utils.data import Dataset


class T91(Dataset):
    def __init__(self, h5_filepath):
        super(T91, self).__init__()
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.file_path = h5_filepath

    def __getitem__(self, item):
        # 1。从文件中读取一个数据（例如，plt.imread）。
        # 2。预处理数据（例如torchvision.Transform）。
        # 3。返回数据对（例如图像和标签）。
        # 这里需要注意的是，第一步：read one data，是一个data
        with h5py.File(self.file_path, 'r') as f:
            lr = f['lr'][item] / 255
            lr = torch.from_numpy(lr).unsqueeze(0)
            hr = f['lr'][item] / 255
            hr = torch.from_numpy(hr).unsqueeze(0)

            return lr, hr

    def __len__(self):
        # 返回数据集的总大小。
        with h5py.File(self.file_path, 'r') as f:
            return len(f['lr'])

