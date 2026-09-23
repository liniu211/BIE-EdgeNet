"""
绝对路径协议的变化检测数据集

适配 E:\\rqx\\dataes 下 WHU-CD / GVLM-CD / LBFD-CD 的数据架构：
    <root_dir>\\<split>\\t1\\        前时相影像
    <root_dir>\\<split>\\t2\\        后时相影像
    <root_dir>\\<split>\\label\\     变化标签
    <root_dir>\\<split>.txt          可选；三列绝对路径索引（t1 t2 label）

与旧 CDDataset（依赖 A/B/label/label_edge/list 协议）并列，互不影响。
索引加载采用自动双模式：存在 <split>.txt 则解析三列绝对路径，否则运行时扫描目录。
label 统一二值化为 0/1，兼容 bool 型 .tif、多通道 png、uint8 png 三种情况。
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation


IMG_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _load_abs_list(root_dir, split):
    """
    入参:
        root_dir: str, 数据集根目录
        split: str, 数据划分名（train / val / test）
    方法:
        优先解析 <root_dir>/<split>.txt 的三列绝对路径（t1 t2 label）；
        若该 txt 不存在，则扫描 <root_dir>/<split>/t1/ 目录，按文件名
        同名规则在 t2/ 与 label/ 下配对，构造三列绝对路径。
    出参:
        list[tuple(str, str, str)]: 每元素为 (t1_path, t2_path, label_path)
        所有路径均经存在性校验，缺失项直接抛 FileNotFoundError
    """
    root = Path(root_dir)
    txt_path = root / f'{split}.txt'
    samples = []

    if txt_path.exists():
        with open(txt_path, 'r', encoding='utf-8') as f:
            for raw in f:
                parts = raw.split()
                if len(parts) < 3:
                    continue
                t1, t2, label = parts[0], parts[1], parts[2]
                for p in (t1, t2, label):
                    if not os.path.exists(p):
                        raise FileNotFoundError(f'索引文件 {txt_path} 引用的路径不存在: {p}')
                samples.append((t1, t2, label))
    else:
        t1_dir = root / split / 't1'
        t2_dir = root / split / 't2'
        label_dir = root / split / 'label'
        if not t1_dir.exists():
            raise FileNotFoundError(f'未找到索引文件 {txt_path}，且扫描目录 {t1_dir} 不存在')

        for t1_path in sorted(t1_dir.iterdir()):
            if t1_path.suffix.lower() not in IMG_EXT:
                continue
            stem = t1_path.name
            t2_path = t2_dir / stem
            label_path = label_dir / stem
            for p in (t1_path, t2_path, label_path):
                if not p.exists():
                    raise FileNotFoundError(f'目录扫描配对缺失: 期望存在 {p}')
            samples.append((str(t1_path), str(t2_path), str(label_path)))

    if len(samples) == 0:
        raise RuntimeError(f'数据集 {root_dir} split={split} 样本数为 0')
    return samples


class CDDatasetAbs(data.Dataset):
    """
    绝对路径协议的变化检测数据集，不依赖 edge 标签。
    """

    def __init__(self, root_dir, img_size, split='train', is_train=True,
                 label_transform=None, to_tensor=True):
        """
        入参:
            root_dir: str, 数据集根目录（含 <split>/t1,t2,label 或 <split>.txt）
            img_size: int, 训练/推理裁剪尺寸
            split: str, 数据划分名（train / val / test）
            is_train: bool, 是否为训练阶段（启用增广）
            label_transform: str|None, 为 'norm' 时将 label 归一为 0/1
            to_tensor: bool, 是否在 __getitem__ 内转为 tensor
        方法:
            预加载全部样本的 (t1, t2, label) 绝对路径三元组；
            按 is_train 选择完整增广或仅 resize 的 CDDataAugmentation。
        出参: 无（构造器）
        """
        super(CDDatasetAbs, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.label_transform = label_transform
        self.to_tensor = to_tensor

        self.samples = _load_abs_list(root_dir, split)
        self.A_size = len(self.samples)

        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __len__(self):
        return self.A_size

    def __getitem__(self, index):
        """
        入参:
            index: int, 样本索引
        方法:
            取第 index 个三元组路径；
            影像统一转 RGB；label 取灰度并二值化为 0/1（>0 即前景），
            兼容 bool 型 .tif / 多通道 png / uint8 png；
            经共享增广器对 (A,B) 与 (L) 做同步变换并 to_tensor/normalize。
        出参:
            dict: {'name': str, 'A': Tensor[3,H,W], 'B': Tensor[3,H,W], 'L': Tensor[1,H,W]}
        """
        t1_path, t2_path, label_path = self.samples[index % self.A_size]
        name = os.path.basename(t1_path)

        img = np.asarray(Image.open(t1_path).convert('RGB'))
        img_B = np.asarray(Image.open(t2_path).convert('RGB'))

        label_arr = np.array(Image.open(label_path))
        if label_arr.ndim == 3:
            label_arr = label_arr[..., 0]
        # 统一二值化：bool 型 / 0-255 / 任意非零值，>0 即前景
        label = (label_arr > 0).astype(np.uint8)

        [img, img_B], [label] = self.augm.transform(
            [img, img_B], [label], to_tensor=self.to_tensor)

        return {'name': name, 'A': img, 'B': img_B, 'L': label}
