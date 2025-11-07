from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from scipy.signal import convolve2d
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import time
from tqdm import tqdm
import json
import pickle
import io
import cv2

import libs.clip
import bisect

# 导入npz数据加载器
try:
    from npz_data_loader import NPZDataLoader
except ImportError:
    # 如果在scripts目录下运行，尝试从上级目录导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from npz_data_loader import NPZDataLoader


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root, need_squeeze=False, full_feature=False, fix_test_order=False):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)
        self.need_squeeze = need_squeeze
        self.full_feature = full_feature
        self.fix_test_order = fix_test_order

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.full_feature:
            z = np.load(os.path.join(self.root, f'{index}.npy'))

            if self.fix_test_order:
                k = self.n_captions[index] - 1
            else:
                k = random.randint(0, self.n_captions[index] - 1)

            test_item = np.load(os.path.join(self.root, f'{index}_{k}.npy'), allow_pickle=True).item()
            token_embedding = test_item['token_embedding']
            token_mask = test_item['token_mask']
            token = test_item['token']
            caption = test_item['promt']
            return z, token_embedding, token_mask, token, caption
        else:
            z = np.load(os.path.join(self.root, f'{index}.npy'))
            k = random.randint(0, self.n_captions[index] - 1)
            c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
            if self.need_squeeze:
                return z, c.squeeze()
            else:
                return z, c


class JDBFeatureDataset(Dataset):
    def __init__(self, root, resolution, llm):
        super().__init__()
        json_path = os.path.join(root,'img_text_pair.jsonl')
        self.img_root = os.path.join(root,'imgs')
        self.feature_root = os.path.join(root,'features')
        self.resolution = resolution
        self.llm = llm
        self.file_list = []
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.file_list.append(json.loads(line)['img_path'])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data_item = self.file_list[idx]
        # Extract filename and replace extension with .npy for all image formats
        filename = data_item.split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        feature_path = os.path.join(self.feature_root, base_name + '.npy')
        img_path = os.path.join(self.img_root, data_item)

        train_item = np.load(feature_path, allow_pickle=True).item()
        pil_image = Image.open(img_path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")


        z = train_item[f'image_latent_{self.resolution}']
        token_embedding = train_item[f'token_embedding_{self.llm}']
        token_mask = train_item[f'token_mask_{self.llm}']
        token = train_item[f'token_{self.llm}']
        caption = train_item['batch_caption']

        img = center_crop_arr(pil_image, image_size=self.resolution)
        img = (img / 127.5 - 1.0).astype(np.float32)
        img = einops.rearrange(img, 'h w c -> c h w')

        # return z, token_embedding, token_mask, token, caption, 0, img, 0, 0
        return z, token_embedding, token_mask, token, caption, img


class JDBFullFeatures(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, train_path, val_path, resolution, llm, cfg=False, p_uncond=None, fix_test_order=False):
        super().__init__()
        print('Prepare dataset...')
        self.resolution = resolution

        self.train = JDBFeatureDataset(train_path, resolution=resolution, llm=llm)
        self.test = MSCOCOFeatureDataset(os.path.join(val_path, 'val'), full_feature=True, fix_test_order=fix_test_order)
        assert len(self.test) == 40504
        
        print('Prepare dataset ok')

        # self.empty_context = np.load(os.path.join(val_path, 'empty_context.npy'), allow_pickle=True).item()

        assert not cfg

        # text embedding extracted by clip
        self.prompts, self.token_embedding, self.token_mask, self.token = [], [], [], []
        for f in sorted(os.listdir(os.path.join(val_path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            vis_item = np.load(os.path.join(val_path, 'run_vis', f), allow_pickle=True).item()
            self.prompts.append(vis_item['promt'])
            self.token_embedding.append(vis_item['token_embedding'])
            self.token_mask.append(vis_item['token_mask'])
            self.token.append(vis_item['token'])
        self.token_embedding = np.array(self.token_embedding)
        self.token_mask = np.array(self.token_mask)
        self.token = np.array(self.token)

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32

    @property
    def fid_stat(self):
        return f'/storage/v-jinpewang/lab_folder/qisheng_data/assets/fid_stats/fid_stats_mscoco256_val.npz'

class TextImageDataset(Dataset):
    def __init__(self, feature_dir, resolution, llm, mode='train', train_img_path=None):
        """
        Args:
            feature_dir: Path to directory containing feature files (.npy for val/run_vis, .npz for train)
            resolution: Image resolution 
            llm: Language model type ('t5')
            mode: 'train', 'val', or 'run_vis'
            train_img_path: Path to training images (only needed for training)
        """
        super().__init__()
        self.feature_dir = feature_dir
        self.resolution = resolution
        self.llm = llm
        self.mode = mode
        self.train_img_path = train_img_path
        
        if self.mode == 'train':
            # 训练模式：使用NPZDataLoader加载npz文件
            print(f"[训练模式] 使用NPZ数据加载器: {feature_dir}")
            self.npz_loader = NPZDataLoader(feature_dir, llm=llm, resolution=resolution)
            self.sample_names = self.npz_loader.get_sample_names()
            print(f"[训练模式] 加载完成，共 {len(self.sample_names)} 个样本")
        else:
            # 验证和run_vis模式：使用原有的npy文件加载方式
            print(f"[{mode}模式] 使用NPY文件加载: {feature_dir}")
            self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
            self.feature_files.sort()  # Ensure consistent ordering
            print(f"[{mode}模式] 加载完成，共 {len(self.feature_files)} 个文件")

    def __len__(self):
        if self.mode == 'train':
            return len(self.sample_names)
        else:
            return len(self.feature_files)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # 训练模式：使用NPZDataLoader加载数据
            sample_name = self.sample_names[idx]
            
            # 从 npz 文件中加载特征数据
            feature_data = self.npz_loader.load_sample(sample_name)
            if feature_data is None:
                raise RuntimeError(f"Failed to load sample {sample_name} from NPZ files")
            
            # 提取特征
            z = feature_data[f'image_latent_{self.resolution}']
            token_embedding = feature_data[f'token_embedding_{self.llm}'][:77]  # 截取前77个token
            token_mask = feature_data[f'token_mask_{self.llm}'][:77]  # 截取前77个token
            token_mask = torch.tensor(token_mask)
            
            # 从输出图像路径加载原始图像
            output_image_path = self.npz_loader.get_output_image_path(sample_name, self.train_img_path)
            
            if output_image_path is None or not os.path.exists(output_image_path):
                # 如果没有路径信息或文件不存在，尝试传统方式
                train_img_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    potential_path = os.path.join(self.train_img_path, sample_name + ext)
                    if os.path.exists(potential_path):
                        train_img_file = potential_path
                        break
                
                if train_img_file is None:
                    raise FileNotFoundError(f"Training image not found for {sample_name} in {self.train_img_path}")
                output_image_path = train_img_file
            
            # 加载和预处理图像
            pil_image = Image.open(output_image_path)
            pil_image.load()
            pil_image = pil_image.convert("RGB")
            
            img = center_crop_arr(pil_image, image_size=self.resolution)
            img = (img / 127.5 - 1.0).astype(np.float32)
            img = einops.rearrange(img, 'h w c -> c h w')
            
            return z, token_embedding, token_mask, img
            
        else:
            # 验证和run_vis模式：使用原有的npy文件加载方式
            feature_file = self.feature_files[idx]
            base_name = os.path.splitext(feature_file)[0]
            feature_path = os.path.join(self.feature_dir, feature_file)
            
            # Load feature data
            feature_data = np.load(feature_path, allow_pickle=True).item()
            
            # Extract token embeddings and mask
            token_embedding = feature_data[f'token_embedding_{self.llm}'][:77]  # 截取前77个token
            token_mask = feature_data[f'token_mask_{self.llm}'][:77]  # 截取前77个token
            token_mask = torch.tensor(token_mask)
            
            # For validation and run_vis, only return token_embedding and token_mask
            return token_embedding, token_mask


class TextImageFeatures(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, train_feature_dir, train_img_path, val_feature_dir, run_vis_feature_dir, resolution, llm, cfg=False, p_uncond=None, fix_test_order=False):
        """
        Args:
            train_feature_dir: Path to training set feature directory
            train_img_path: Path to training images  
            val_feature_dir: Path to validation set feature directory
            run_vis_feature_dir: Path to run_vis feature directory
            resolution: Image resolution
            llm: Language model type ('t5')
        """
        super().__init__()
        print('Prepare dataset...')
        self.resolution = resolution

        # Create training dataset
        self.train = TextImageDataset(
            feature_dir=train_feature_dir, 
            resolution=resolution, 
            llm=llm, 
            mode='train',
            train_img_path=train_img_path
        )
        
        # Create validation dataset
        self.test = TextImageDataset(
            feature_dir=val_feature_dir,
            resolution=resolution,
            llm=llm,
            mode='val'
        )
        
        # Create run_vis dataset and extract embeddings
        run_vis_dataset = TextImageDataset(
            feature_dir=run_vis_feature_dir,
            resolution=resolution,
            llm=llm,
            mode='run_vis'
        )
        
        print('Prepare dataset ok')

        assert not cfg

        # Extract token embeddings and masks from run_vis dataset
        self.token_embedding = []
        self.token_mask = []
        
        for i in range(len(run_vis_dataset)):
            token_embedding, token_mask = run_vis_dataset[i]
            self.token_embedding.append(token_embedding)
            self.token_mask.append(token_mask.numpy() if isinstance(token_mask, torch.Tensor) else token_mask)
            
        self.token_embedding = np.array(self.token_embedding)
        self.token_mask = np.array(self.token_mask)

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32
    
    @property
    def fid_stat(self):
        return f'/storage/v-jinpewang/lab_folder/qisheng_data/assets/fid_stats/fid_stats_mscoco256_val.npz'


def get_dataset(name, **kwargs):
    if name == 'JDB_demo_features':
        return JDBFullFeatures(**kwargs)
    elif name == 'textimage_features':
        return TextImageFeatures(**kwargs)
    else:
        raise NotImplementedError(name)


