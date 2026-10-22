"""项目适配器：已有PNG配对 -> 官方训练程序的lq/gt字段。

特意不引入官方 Dataset_GaussianDenoising，避免再次添加高斯噪声。
不调用random_crop、augment、mixup或任何噪声生成函数。
"""

from torch.utils.data import Dataset
from paired_dataset import DenoisingDataset


class FixedPairedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dataset = DenoisingDataset(opt["data_root"], opt["dataset"], opt["phase"])
        self.fingerprint = self.dataset.fingerprint
        self.source_ids = self.dataset.source_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        noisy, gt, group = self.dataset[index]
        noisy_path, gt_path, _ = self.dataset.samples[index]
        return {"lq": noisy, "gt": gt, "group": group,
                "lq_path": str(noisy_path), "gt_path": str(gt_path)}
