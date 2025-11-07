#!/usr/bin/env python3
"""
NPZ格式数据加载器 - 直接从批量保存的npz文件加载数据
专用于2D VAE格式，适配extract_train_feature_2D.py脚本
兼容原npy格式的接口，无需转换
"""

import os
import numpy as np
from typing import Dict, List, Optional
import glob

class NPZDataLoader:
    """
    高效加载批量保存的npz文件 (仅支持2D VAE格式)
    
    用法示例:
        loader = NPZDataLoader('/path/to/train_features/train_01')
        
        # 方式1: 按名称加载单个样本
        data = loader.load_sample('image_001')
        
        # 方式2: 批量加载
        batch_data = loader.load_batch(['image_001', 'image_002', 'image_003'])
        
        # 方式3: 迭代所有样本
        for sample_name, data in loader.iter_samples():
            print(sample_name, data.keys())
            
        # 方式4: 获取图像路径信息
        input_path = loader.get_input_image_path('image_001', '/data/input')
        output_path = loader.get_output_image_path('image_001', '/data/output')
    """
    
    def __init__(self, data_dir: str, llm=None, resolution=None):
        self.data_dir = data_dir
        self._llm = llm  # 用户指定的llm（可选）
        self._resolution = resolution  # 用户指定的resolution（可选）
        self.llm = None  # 实际使用的llm（从npz自动检测）
        self.resolution = None  # 实际使用的resolution（从npz自动检测）
        self._index = None
        self._build_index()
    
    def _build_index(self):
        """构建样本名称到npz文件的映射，并自动检测元数据"""
        self._index = {}  # {sample_name: (npz_file, index_in_batch)}
        self._input_path_index = {}  # {sample_name: input_relative_path}
        self._output_path_index = {}  # {sample_name: output_relative_path}
        
        # 查找所有npz文件
        npz_files = glob.glob(os.path.join(self.data_dir, 'batch_*.npz'))
        
        print(f"找到 {len(npz_files)} 个npz批量文件")
        
        # 从第一个文件读取元数据
        metadata_loaded = False
        
        for npz_file in npz_files:
            try:
                # 读取文件
                data = np.load(npz_file, allow_pickle=True)
                sample_names = data['sample_names']
                
                # 读取图像相对路径（优先使用新的字段名）
                input_relative_paths = None
                output_relative_paths = None
                
                if 'input_image_relative_paths' in data:
                    input_relative_paths = data['input_image_relative_paths']
                if 'output_image_relative_paths' in data:
                    output_relative_paths = data['output_image_relative_paths']
                    
                # 兼容旧的字段名
                if input_relative_paths is None and 'image_relative_paths' in data:
                    input_relative_paths = data['image_relative_paths']
                    output_relative_paths = data['image_relative_paths']
                
                # 第一次读取时，获取元数据
                if not metadata_loaded:
                    # 验证是否为2D VAE格式
                    vae_type = data.get('vae_type', '2D')
                    if isinstance(vae_type, (bytes, np.ndarray)):
                        vae_type = str(vae_type).strip("b'\"")
                    
                    if vae_type != '2D':
                        print(f"警告：检测到非2D VAE格式: {vae_type}，将强制使用2D模式")
                    
                    # 尝试读取元数据（如果有的话）
                    if 'llm' in data:
                        self.llm = str(data['llm']).strip("b'\"")
                    if 'resolution' in data:
                        self.resolution = int(data['resolution']) if not isinstance(data['resolution'], np.ndarray) else int(data['resolution'].item())
                    
                    # 如果用户指定了参数，使用用户指定的（优先级更高）
                    if self._llm is not None:
                        self.llm = self._llm
                    if self._resolution is not None:
                        self.resolution = self._resolution
                    
                    # 如果仍然没有，使用默认值
                    if self.llm is None:
                        self.llm = 't5'
                        print(f"警告：未找到LLM元数据，使用默认值: {self.llm}")
                    if self.resolution is None:
                        self.resolution = 256
                        print(f"警告：未找到分辨率元数据，使用默认值: {self.resolution}")
                    
                    print(f"元数据：LLM={self.llm}, Resolution={self.resolution}, VAE=2D")
                    metadata_loaded = True
                
                # 建立索引
                for i, name in enumerate(sample_names):
                    self._index[name] = (npz_file, i)
                    # 保存输入和输出图像的相对路径
                    if input_relative_paths is not None:
                        self._input_path_index[name] = input_relative_paths[i]
                    if output_relative_paths is not None:
                        self._output_path_index[name] = output_relative_paths[i]
                
                data.close()
            except Exception as e:
                print(f"警告：无法索引 {npz_file}: {e}")
        
        print(f"索引完成，共 {len(self._index)} 个样本")
    
    def load_sample(self, sample_name: str) -> Optional[Dict]:
        """
        加载单个样本 (2D VAE格式)
        
        返回格式:
        {
            'image_latent_256': ndarray,      # 图像 latent 特征 [8, 32, 32]
            'token_embedding_t5': ndarray,    # token 嵌入特征 [576, 2048]
            'token_mask_t5': ndarray,         # token 注意力掩码 [576]
        }
        """
        if sample_name not in self._index:
            print(f"警告：样本 {sample_name} 不存在")
            return None
        
        npz_file, index = self._index[sample_name]
        
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # 只支持2D VAE格式
            if 'moments' not in data:
                print(f"错误：样本 {sample_name} 不是2D VAE格式（缺少'moments'字段）")
                data.close()
                return None
            
            # 返回2D VAE数据
            sample_data = {
                f'image_latent_{self.resolution}': data['moments'][index],
                f'token_embedding_{self.llm}': data['embeddings'][index],
                f'token_mask_{self.llm}': data['masks'][index],
            }
            
            data.close()
            return sample_data
        
        except Exception as e:
            print(f"错误加载样本 {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_batch(self, sample_names: List[str]) -> Dict[str, Dict]:
        """
        批量加载多个样本（更高效） - 2D VAE格式
        
        返回: {sample_name: sample_data, ...}
        """
        # 按npz文件分组
        file_groups = {}
        for name in sample_names:
            if name in self._index:
                npz_file, index = self._index[name]
                if npz_file not in file_groups:
                    file_groups[npz_file] = []
                file_groups[npz_file].append((name, index))
        
        # 批量加载
        results = {}
        for npz_file, items in file_groups.items():
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # 只支持2D VAE格式
                if 'moments' not in data:
                    print(f"警告：文件 {npz_file} 不是2D VAE格式，跳过")
                    data.close()
                    continue
                
                for name, index in items:
                    results[name] = {
                        f'image_latent_{self.resolution}': data['moments'][index],
                        f'token_embedding_{self.llm}': data['embeddings'][index],
                        f'token_mask_{self.llm}': data['masks'][index],
                    }
                
                data.close()
            except Exception as e:
                print(f"错误加载批次文件 {npz_file}: {e}")
        
        return results
    
    def iter_samples(self, batch_size: int = 32):
        """
        迭代所有样本（内存友好） - 2D VAE格式
        
        Args:
            batch_size: 每次从同一个npz文件读取的样本数
        
        Yields:
            (sample_name, sample_data)
        """
        # 按npz文件分组
        file_groups = {}
        for name, (npz_file, index) in self._index.items():
            if npz_file not in file_groups:
                file_groups[npz_file] = []
            file_groups[npz_file].append((name, index))
        
        # 遍历每个npz文件
        for npz_file, items in file_groups.items():
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # 只支持2D VAE格式
                if 'moments' not in data:
                    print(f"警告：文件 {npz_file} 不是2D VAE格式，跳过")
                    data.close()
                    continue
                
                # 批量读取
                for i in range(0, len(items), batch_size):
                    batch_items = items[i:i+batch_size]
                    
                    for name, index in batch_items:
                        sample_data = {
                            f'image_latent_{self.resolution}': data['moments'][index],
                            f'token_embedding_{self.llm}': data['embeddings'][index],
                            f'token_mask_{self.llm}': data['masks'][index],
                        }
                        yield name, sample_data
                
                data.close()
            except Exception as e:
                print(f"错误读取 {npz_file}: {e}")
    
    def get_sample_names(self) -> List[str]:
        """获取所有样本名称"""
        return list(self._index.keys())
    
    def get_input_image_path(self, sample_name: str, image_root: str = None) -> Optional[str]:
        """
        获取样本对应的输入图像路径
        
        Args:
            sample_name: 样本名称
            image_root: 输入图像根目录（如果提供，会返回绝对路径）
        
        Returns:
            输入图像相对路径或绝对路径，如果没有保存路径信息则返回None
        """
        if sample_name not in self._input_path_index:
            return None
        
        rel_path = self._input_path_index[sample_name]
        
        if image_root is not None:
            # 返回绝对路径
            return os.path.join(image_root, rel_path)
        else:
            # 返回相对路径
            return rel_path
    
    def get_output_image_path(self, sample_name: str, image_root: str = None) -> Optional[str]:
        """
        获取样本对应的输出图像路径
        
        Args:
            sample_name: 样本名称
            image_root: 输出图像根目录（如果提供，会返回绝对路径）
        
        Returns:
            输出图像相对路径或绝对路径，如果没有保存路径信息则返回None
        """
        if sample_name not in self._output_path_index:
            return None
        
        rel_path = self._output_path_index[sample_name]
        
        if image_root is not None:
            # 返回绝对路径
            return os.path.join(image_root, rel_path)
        else:
            # 返回相对路径
            return rel_path
    
    def get_image_path(self, sample_name: str, image_root: str = None) -> Optional[str]:
        """
        兼容旧接口，返回输入图像路径
        
        Args:
            sample_name: 样本名称
            image_root: 图像根目录（如果提供，会返回绝对路径）
        
        Returns:
            图像相对路径或绝对路径，如果没有保存路径信息则返回None
        """
        return self.get_input_image_path(sample_name, image_root)
    
    def __len__(self):
        """样本总数"""
        return len(self._index)
    
    def __contains__(self, sample_name):
        """检查样本是否存在"""
        return sample_name in self._index


# 向后兼容：模拟原npy加载接口
def load_feature(data_dir: str, sample_name: str) -> Optional[Dict]:
    """
    加载单个样本特征（兼容原npy格式的代码）
    
    自动检测是npz批量格式还是npy单文件格式
    只支持2D VAE格式
    """
    # 先尝试npy格式
    npy_file = os.path.join(data_dir, f'{sample_name}.npy')
    if os.path.exists(npy_file):
        data = np.load(npy_file, allow_pickle=True).item()
        # 检查是否为2D VAE格式
        if 'image_latent_256' not in data and 'image_latent_512' not in data:
            print(f"警告：npy文件 {npy_file} 不是2D VAE格式")
            return None
        return data
    
    # 如果不存在npy，尝试从npz加载
    loader = NPZDataLoader(data_dir)
    return loader.load_sample(sample_name)


# 使用示例
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python npz_data_loader.py <data_dir> [sample_name]")
        print("示例: python npz_data_loader.py /data/train_features/train_00 000001_00000303")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # 创建加载器
    loader = NPZDataLoader(data_dir)
    print(f"数据目录: {data_dir}")
    print(f"总样本数: {len(loader)}")
    print(f"LLM类型: {loader.llm}")
    print(f"分辨率: {loader.resolution}")
    print("VAE类型: 2D")
    
    if len(sys.argv) >= 3:
        # 加载指定样本
        sample_name = sys.argv[2]
        data = loader.load_sample(sample_name)
        if data:
            print(f"\n样本 {sample_name}:")
            for key, value in data.items():
                print(f"  {key}: {value.shape} {value.dtype}")
            
            # 显示路径信息
            input_path = loader.get_input_image_path(sample_name)
            output_path = loader.get_output_image_path(sample_name)
            if input_path:
                print(f"  输入图像路径: {input_path}")
            if output_path:
                print(f"  输出图像路径: {output_path}")
    else:
        # 显示前5个样本
        print("\n前5个样本:")
        for i, (name, data) in enumerate(loader.iter_samples()):
            print(f"{i+1}. {name}")
            for key, value in data.items():
                print(f"    {key}: {value.shape}")
            if i >= 4:
                break
