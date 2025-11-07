"""
This file is used to extract feature of the demo training data.
Multi-GPU parallel version using Accelerate.
"""

import os
import shutil
import sys
# 往上两级到 Crossflow 目录，以便访问 libs 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import einops
import random
import json
import time
import re
import libs.autoencoder
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object

print("sys.path:",sys.path)
original_sys_path = sys.path.copy()
crossflow_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
janus_dir = os.path.join(crossflow_parent_dir, "Janus")
# sys.path.insert(0, os.path.abspath(janus_dir))
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
sys.path = original_sys_path  # 直接恢复

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def extract_number_key(filename):
    """
    从文件名中提取数字部分作为匹配键
    例如：'000001_00000303_textbox.png' -> '000001_00000303'
          '000001_00000303_edit.png' -> '000001_00000303'
    
    支持多种格式：
    - 连续数字和下划线组合：'000001_00000303'
    - 纯数字：'123456'
    """
    basename = os.path.splitext(filename)[0]  # 移除扩展名
    
    # 尝试匹配连续的数字和下划线模式（最常见）
    # 例如：000001_00000303
    match = re.search(r'^(\d+(?:_\d+)*)', basename)
    if match:
        return match.group(1)
    
    # 如果没找到，尝试提取所有数字部分并用下划线连接
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return '_'.join(numbers)
    
    # 如果都没有，返回原始basename
    return basename

def center_crop_arr(pil_image, image_size):
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


class ImageFeatureDataset(Dataset):
    """自定义数据集类，用于并行加载和处理图像
    
    支持从两个不同的路径加载图像：
    - input_image_path: 用于提取embeddings和masks（输入图像）
    - output_image_path: 用于提取moments（输出图像）
    
    输入和输出图像通过文件名中的数字部分进行匹配，例如：
    - 输入：000001_00000303_textbox.png
    - 输出：000001_00000303_edit.png
    匹配键：000001_00000303
    """
    def __init__(self, input_image_path, output_image_path, save_dir=None, skip_processed=True, recursive=True):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        
        scan_mode = "递归扫描（包括子文件夹）" if recursive else "仅扫描顶层目录"
        print(f"[数据集] 开始{scan_mode}:")
        print(f"  - 输入图像路径（embeddings/masks）: {input_image_path}")
        print(f"  - 输出图像路径（moments）: {output_image_path}")
        scan_start = time.time()
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Step 1: 扫描输入图像路径
        all_input_files = []
        if recursive:
            for root, dirs, files in os.walk(input_image_path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in valid_extensions:
                        rel_path = os.path.relpath(os.path.join(root, filename), input_image_path)
                        all_input_files.append(rel_path)
        else:
            with os.scandir(input_image_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in valid_extensions:
                            all_input_files.append(entry.name)
        
        all_input_files.sort()
        print(f"[数据集] 输入路径扫描完成，共找到 {len(all_input_files)} 张图片")
        
        # Step 2: 扫描输出图像路径并建立数字键映射
        output_map = {}  # 数字键 -> 输出图像相对路径
        if recursive:
            for root, dirs, files in os.walk(output_image_path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in valid_extensions:
                        rel_path = os.path.relpath(os.path.join(root, filename), output_image_path)
                        # 提取数字键
                        number_key = extract_number_key(os.path.basename(filename))
                        output_map[number_key] = rel_path
        else:
            with os.scandir(output_image_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in valid_extensions:
                            number_key = extract_number_key(entry.name)
                            output_map[number_key] = entry.name
        
        print(f"[数据集] 输出路径扫描完成，共找到 {len(output_map)} 张图片")
        
        # Step 3: 建立输入输出映射关系
        self.paired_files = []  # [(input_rel_path, output_rel_path, number_key), ...]
        missing_output = []
        
        for input_file in all_input_files:
            # 提取输入文件的数字键
            input_basename = os.path.basename(input_file)
            number_key = extract_number_key(input_basename)
            
            # 查找对应的输出文件
            if number_key in output_map:
                output_file = output_map[number_key]
                self.paired_files.append((input_file, output_file, number_key))
            else:
                missing_output.append((input_file, number_key))
        
        if missing_output:
            print(f"[数据集] 警告：{len(missing_output)} 张输入图像没有找到对应的输出图像")
            if len(missing_output) <= 10:
                for inp, key in missing_output[:10]:
                    print(f"  - 输入: {inp}, 数字键: {key}")
        
        print(f"[数据集] 成功配对 {len(self.paired_files)} 对图像")
        print(f"[数据集] 扫描完成，耗时: {time.time()-scan_start:.2f}秒")
        
        # 如果需要跳过已处理的图片
        if skip_processed and save_dir and os.path.exists(save_dir):
            print(f"[数据集] 开始检查已处理文件: {save_dir}")
            check_start = time.time()
            
            # 检查已处理的文件（检查NPY和NPZ格式）
            processed_keys = set()  # 使用数字键而不是文件名
            
            # 方式1：检查NPY文件
            with os.scandir(save_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.npy'):
                        # 从npy文件名提取数字键
                        number_key = extract_number_key(entry.name)
                        processed_keys.add(number_key)
            
            # 方式2：检查NPZ文件中的sample_names
            npz_files = [f for f in os.listdir(save_dir) if f.startswith('batch_') and f.endswith('.npz')]
            for npz_file in npz_files:
                try:
                    npz_data = np.load(os.path.join(save_dir, npz_file), allow_pickle=True)
                    if 'sample_names' in npz_data:
                        for sample_name in npz_data['sample_names']:
                            number_key = extract_number_key(sample_name)
                            processed_keys.add(number_key)
                    npz_data.close()
                except:
                    pass
            
            print(f"[数据集] 检查完成，已处理 {len(processed_keys)} 个文件，耗时: {time.time()-check_start:.2f}秒")
            
            # 过滤出未处理的图片对
            filter_start = time.time()
            filtered_pairs = []
            for input_file, output_file, number_key in self.paired_files:
                if number_key not in processed_keys:
                    filtered_pairs.append((input_file, output_file, number_key))
            
            self.paired_files = filtered_pairs
            print(f"[数据集] 过滤完成，待处理: {len(self.paired_files)} 对，耗时: {time.time()-filter_start:.2f}秒")
        
        print(f"[数据集] 最终待处理图像对数量: {len(self.paired_files)}")
    
    def __len__(self):
        return len(self.paired_files)
    
    def __getitem__(self, idx):
        input_rel_path, output_rel_path, number_key = self.paired_files[idx]
        
        # 构建完整路径
        input_image_path = os.path.join(self.input_image_path, input_rel_path)
        output_image_path = os.path.join(self.output_image_path, output_rel_path)
        
        try:
            # 加载输入图像（用于Janus提取embeddings/masks）
            # 注意：这里不做预处理，只是记录路径，实际加载在batch处理时进行
            
            # 加载输出图像（用于autoencoder提取moments）
            with Image.open(output_image_path) as pil_image:
                # 直接转换为RGB，避免后续处理
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                
                # 处理图像 - 只处理256大小
                image_256 = center_crop_arr(pil_image, image_size=256)
            
            # 归一化并转换格式
            image_256 = (image_256 / 127.5 - 1.0).astype(np.float32)
            image_256 = einops.rearrange(image_256, 'h w c -> c h w')
            
            return {
                'image_256': torch.from_numpy(image_256),  # 来自output_image_path，用于moments
                'input_image_path': input_image_path,      # 用于embeddings/masks
                'output_image_path': output_image_path,    # 用于moments
                'img_name': number_key,  # 使用数字键作为样本名
                'success': True
            }
        except Exception as e:
            # 返回一个失败标记
            return {
                'image_256': torch.zeros((3, 256, 256), dtype=torch.float32),
                'input_image_path': '',
                'output_image_path': '',
                'img_name': number_key,  # 使用数字键
                'success': False
            }


def main():
    """
    主函数 - 自动增量处理，跳过已处理的图片
    
    配置参数通过环境变量传入:
        BATCH_SIZE: 批次大小 (默认: 1024, 针对H100 80GB优化)
        INPUT_IMAGE_PATH: 输入图像路径（用于提取embeddings和masks）
        OUTPUT_IMAGE_PATH: 输出图像路径（用于提取moments）
        SAVE_DIR: 保存目录
        MODEL_PATH: Janus模型路径
        AUTOENCODER_PATH: Autoencoder模型路径
        NUM_WORKERS: DataLoader工作进程数
        PREFETCH_FACTOR: 预取因子
        RECURSIVE_SCAN: 是否递归扫描子文件夹
    """
    # 从环境变量读取配置参数
    bz = int(os.getenv('BATCH_SIZE', '1024'))
    input_image_path = os.getenv('INPUT_IMAGE_PATH', '/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/trainset/train_00')
    output_image_path = os.getenv('OUTPUT_IMAGE_PATH', '/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/trainset/train_00_output')
    save_dir = os.getenv('SAVE_DIR', '/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/train_features_2D/train_00')
    model_path = os.getenv('MODEL_PATH', 'deepseek-ai/Janus-Pro-1B')
    autoencoder_path = os.getenv('AUTOENCODER_PATH', '/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth')
    num_workers = int(os.getenv('NUM_WORKERS', '16'))
    prefetch_factor = int(os.getenv('PREFETCH_FACTOR', '8'))
    recursive_scan = os.getenv('RECURSIVE_SCAN', 'true').lower() == 'true'
    # 初始化Accelerator，支持多GPU并行
    accelerator = Accelerator()
    device = accelerator.device
    
    # 只在主进程打印信息和创建文件夹
    if accelerator.is_main_process:
        print(f"使用 {accelerator.num_processes} 个GPU进行并行处理")
        print(f"当前进程设备: {device}")
        print(f"NCCL Backend: {torch.distributed.is_nccl_available()}")
        print(f"增量处理模式：自动跳过已处理的图片")
        print(f"\n配置参数:")
        print(f"  批次大小: {bz}")
        print(f"  输入图像路径（embeddings/masks）: {input_image_path}")
        print(f"  输出图像路径（moments）: {output_image_path}")
        print(f"  保存路径: {save_dir}")
        print(f"  递归扫描: {recursive_scan}")
        print(f"  工作进程: {num_workers}")
    
    # 打印每个进程的信息（用于调试）
    print(f"[Rank {accelerator.process_index}] 初始化完成，设备: {device}")
    
    # 只在主进程创建保存目录（如果不存在）
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建保存目录: {save_dir}")
        else:
            print(f"保存目录已存在，将进行增量处理: {save_dir}")
    accelerator.wait_for_everyone()  # 等待主进程检查/创建完文件夹
    
    # 加载模型
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().eval().to(device)
    
    # 加载autoencoder
    autoencoder = libs.autoencoder.get_model(autoencoder_path)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    autoencoder = autoencoder.to(device)
    
    # 创建数据集和数据加载器（支持增量处理）
    # recursive=True: 递归扫描子文件夹（稍慢，但支持任意目录结构）
    # recursive=False: 只扫描顶层（更快，适合图片都在同一层的情况）
    dataset = ImageFeatureDataset(
        input_image_path,   # 用于提取embeddings和masks
        output_image_path,  # 用于提取moments
        save_dir=save_dir, 
        skip_processed=True,
        recursive=recursive_scan  # 从环境变量读取
    )
    
    # 如果没有待处理的样本，直接退出
    if len(dataset) == 0:
        if accelerator.is_main_process:
            print("没有需要处理的新图片，程序退出。")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=bz, 
        shuffle=False,
        num_workers=num_workers,  # 从环境变量读取
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程，减少创建销毁开销
        prefetch_factor=prefetch_factor  # 从环境变量读取
    )
    
    if accelerator.is_main_process:
        print(f"总样本数: {len(dataset)}")
        print(f"每个GPU批次大小: {bz}")
        print(f"总批次数: {len(dataloader)}")
    
    # 使用accelerator准备数据加载器（模型已经手动移到设备上）
    # 注意：对于推理任务，不需要用DDP包装模型，只需要分布式的dataloader
    dataloader = accelerator.prepare(dataloader)
    
    # 推理任务直接使用模型，不需要unwrap
    unwrapped_vl_gpt = vl_gpt
    
    idx = 0
    failed_samples = []
    processing_error = False  # 用于标记是否发生错误
    
    # 使用tqdm显示进度（只在主进程显示）
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc="处理批次")
    
    # 在开始处理前，确保所有进程都准备好了
    if accelerator.is_main_process:
        print(f"\n[主进程] 开始处理，等待所有进程准备...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[主进程] 所有进程已就绪，开始处理批次\n")
    
    # 整个推理过程禁用梯度计算，避免内存泄漏
    with torch.no_grad():
      try:
        for batch_idx, batch in enumerate(dataloader):
            # 记录每个批次的开始时间
            batch_start_time = time.time()
            
            # 每个批次开始时打印进程信息（调试用）
            if batch_idx % 5 == 0:
                print(f"[Rank {accelerator.process_index}] 开始处理批次 {batch_idx}")
            
            # 过滤失败的样本
            valid_mask = batch['success']
            
            # 记录失败的样本
            failed_names = [n for n, v in zip(batch['img_name'], valid_mask) if not v]
            if failed_names:
                failed_samples.extend(failed_names)
            
            # 只处理成功加载的样本
            batch_img_256 = batch['image_256'][valid_mask]
            batch_input_image_paths = [p for p, v in zip(batch['input_image_path'], valid_mask) if v]
            batch_output_image_paths = [p for p, v in zip(batch['output_image_path'], valid_mask) if v]
            batch_names = [n for n, v in zip(batch['img_name'], valid_mask) if v]  # 现在是数字键
            
            # 计算输入图像相对路径（相对于input_image_path，用于embeddings/masks）
            batch_input_relative_paths = []
            for img_path in batch_input_image_paths:
                try:
                    rel_path = os.path.relpath(img_path, input_image_path)
                    batch_input_relative_paths.append(rel_path)
                except:
                    batch_input_relative_paths.append(os.path.basename(img_path))
            
            # 计算输出图像相对路径（相对于output_image_path，用于moments）
            batch_output_relative_paths = []
            for img_path in batch_output_image_paths:
                try:
                    rel_path = os.path.relpath(img_path, output_image_path)
                    batch_output_relative_paths.append(rel_path)
                except:
                    batch_output_relative_paths.append(os.path.basename(img_path))
            
            # 确保所有进程都走相同的代码路径
            if len(batch_names) == 0:
                # 确保CUDA操作完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 更新进度条但不提前continue，保持所有进程同步
                if accelerator.is_main_process:
                    progress_bar.update(1)
                
                # 添加同步点，确保所有进程都到达这里
                print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} 无有效样本，等待同步")
                accelerator.wait_for_everyone()
                continue
            
            # 处理Janus embeddings - 批量处理版本
            batch_embeddings = []
            batch_attention_masks = []
            
            # 打印当前批次的样本数（调试用）
            if batch_idx % 5 == 0:
                print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} 有 {len(batch_input_image_paths)} 个有效样本，开始批量处理Janus...")
            
            # 记录Janus处理时间
            janus_start_time = time.time()
            
            try:
                # 真正的批量处理 - 一次性加载和预处理所有图像
                question = ""
                
                # Step 1: 批量加载所有PIL图像（并行I/O）- 从input_image_path加载
                load_start = time.time()
                all_pil_images = []
                for image_path in batch_input_image_paths:
                    try:
                        pil_img = Image.open(image_path).convert('RGB')
                        all_pil_images.append(pil_img)
                    except Exception as e:
                        print(f"[Rank {accelerator.process_index}] 警告：加载输入图像失败 {image_path}: {e}")
                        # 创建一个空白图像作为占位
                        all_pil_images.append(Image.new('RGB', (384, 384), color='black'))
                load_time = time.time() - load_start
                
                # Step 2: 批量图像预处理（一次性处理所有图像）
                preprocess_start = time.time()
                images_outputs = vl_chat_processor.image_processor(all_pil_images, return_tensors="pt")
                batched_pixel_values = images_outputs.pixel_values.to(device)  # [batch_size, 3, H, W]
                preprocess_time = time.time() - preprocess_start
                
                # Step 3: 准备批量输入（文本部分）
                text_start = time.time()
                # 创建统一的conversation格式（所有图像使用相同的prompt）
                sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=[
                        {"role": "<|User|>", "content": f"<image_placeholder>\n{question}"},
                        {"role": "<|Assistant|>", "content": ""},
                    ],
                    sft_format=vl_chat_processor.sft_format,
                    system_prompt=vl_chat_processor.system_prompt,
                )
                
                # tokenize（所有样本使用相同的token）
                input_ids = vl_chat_processor.tokenizer.encode(sft_format)
                input_ids = torch.LongTensor(input_ids)
                
                # 为批次中的每个样本复制input_ids
                batch_size = len(all_pil_images)
                batched_input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
                batched_attention_mask = torch.ones_like(batched_input_ids)
                
                # 创建image masks
                image_token_mask = batched_input_ids == vl_chat_processor.image_id
                batched_images_seq_mask = image_token_mask
                
                # 假设每个图像有576个token（根据Janus的配置）
                num_image_tokens = 576
                batched_images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens)).bool()
                batched_images_emb_mask[:, :, :num_image_tokens] = True
                
                # 调整pixel_values形状以匹配期望的 [batch_size, n_images, 3, H, W]
                batched_pixel_values = batched_pixel_values.unsqueeze(1)  # [batch_size, 1, 3, H, W]
                
                text_time = time.time() - text_start
                
                # Step 4: 批量编码
                encode_start = time.time()
                with torch.no_grad():
                    inputs_embeds = unwrapped_vl_gpt.prepare_inputs_embeds(
                        input_ids=batched_input_ids.to(device),
                        pixel_values=batched_pixel_values,
                        images_seq_mask=batched_images_seq_mask.to(device),
                        images_emb_mask=batched_images_emb_mask.to(device)
                    )
                encode_time = time.time() - encode_start
                
                # 打印详细的时间分解
                if batch_idx % 5 == 0:
                    total_prep = load_time + preprocess_time + text_time
                    print(f"[Rank {accelerator.process_index}] 时间分解 - 图像加载: {load_time:.2f}s, 预处理: {preprocess_time:.2f}s, 文本: {text_time:.3f}s, 编码: {encode_time:.2f}s")
                    print(f"[Rank {accelerator.process_index}] 批量形状 - pixel_values: {batched_pixel_values.shape}, inputs_embeds: {inputs_embeds.shape}")
                
                # inputs_embeds.shape: [batch_size, 576, 2048]
                # 批量转换到CPU（比逐个转换快得多）
                inputs_embeds_cpu = inputs_embeds.detach().cpu().float().numpy()
                
                # 将批量结果拆分为单个样本
                for i in range(inputs_embeds_cpu.shape[0]):
                    final_tensor = inputs_embeds_cpu[i]  # [576, 2048]
                    batch_embeddings.append(final_tensor)
                    attention_mask = [1] * 576
                    batch_attention_masks.append(attention_mask)
                
                janus_time = time.time() - janus_start_time
                if batch_idx % 5 == 0:
                    print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} Janus批量处理耗时: {janus_time:.2f}秒 (平均 {janus_time/len(batch_input_image_paths):.3f}秒/张)")
            
            except Exception as e:
                # 如果批量处理失败，回退到逐个处理
                print(f"[Rank {accelerator.process_index}] 警告：批次 {batch_idx} 批量处理失败: {e}，回退到逐个处理")
                import traceback
                traceback.print_exc()
                
                batch_embeddings = []
                batch_attention_masks = []
                question = ""
                
                for img_idx, image_path in enumerate(batch_input_image_paths):
                    try:
                        conversation = [
                            {
                                "role": "<|User|>",
                                "content": f"<image_placeholder>\n{question}",
                                "images": [image_path],
                            },
                            {"role": "<|Assistant|>", "content": ""},
                        ]
                        
                        pil_images = load_pil_images(conversation)
                        prepare_inputs = vl_chat_processor(
                            conversations=conversation, images=pil_images, force_batchify=True
                        ).to(device)
                        
                        with torch.no_grad():
                            inputs_embeds = unwrapped_vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                        
                        final_tensor = inputs_embeds.squeeze(dim=0)
                        batch_embeddings.append(final_tensor.detach().cpu().float().numpy())
                        attention_mask = [1] * 576
                        batch_attention_masks.append(attention_mask)
                    
                    except Exception as e2:
                        print(f"[Rank {accelerator.process_index}] 错误：样本 {img_idx} 处理失败: {e2}")
                        batch_embeddings.append(np.zeros((576, 2048), dtype=np.float32))
                        batch_attention_masks.append([1] * 576)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                janus_time = time.time() - janus_start_time
                if batch_idx % 5 == 0:
                    print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} Janus逐个处理耗时: {janus_time:.2f}秒")
            
            # 记录Autoencoder处理时间
            autoencoder_start_time = time.time()
            
            # 处理autoencoder编码 - 只处理256
            with torch.no_grad():
                moments_256 = autoencoder(batch_img_256, fn='encode_moments')
                if moments_256.dim() > 3:
                    moments_256 = moments_256.squeeze(0)
                moments_256 = moments_256.detach().cpu().numpy()
            
            # 确保CUDA操作完成，避免异步操作导致的问题
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            autoencoder_time = time.time() - autoencoder_start_time
            if batch_idx % 5 == 0:
                print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} Autoencoder处理耗时: {autoencoder_time:.2f}秒")
            
            # 保存特征 - 批量保存npz优化版（2D VAE格式）
            save_start = time.time()
            
            process_id = accelerator.process_index
            saved_count = 0
            
            # 策略：先批量保存npz（极快），后续可以用单独脚本转换回npy
            batch_file = os.path.join(save_dir, f'batch_{batch_idx}_rank{process_id}.npz')
            
            if not os.path.exists(batch_file):
                # 准备批量数据（2D VAE格式，与dataset.py兼容）
                # 确定LLM类型：根据使用的模型
                llm_type = 't5'  # 与配置文件保持一致
                
                batch_data = {
                    'sample_names': batch_names,
                    'input_image_relative_paths': batch_input_relative_paths,    # 输入图像相对路径（用于embeddings/masks）
                    'output_image_relative_paths': batch_output_relative_paths,  # 输出图像相对路径（用于moments）
                    'vae_type': '2D',  # 标记为2D VAE格式
                    'llm': llm_type,   # 保存LLM类型
                    'resolution': 256, # 保存分辨率
                }
                
                # 收集所有数据
                all_moments = []
                all_embeddings = []
                all_masks = []
                
                for mt_256, te_t, tm_t in zip(
                    moments_256,  
                    batch_embeddings, 
                    batch_attention_masks):
                    
                    # 验证形状（只在调试时使用）
                    # assert mt_256.shape == (8, 32, 32)
                    # assert te_t.shape == (576, 2048)
                    
                    all_moments.append(mt_256)
                    all_embeddings.append(te_t)
                    all_masks.append(np.array(tm_t))
                
                # 堆叠为数组（更高效）
                batch_data['moments'] = np.stack(all_moments)  # [batch_size, 8, 32, 32]
                batch_data['embeddings'] = np.stack(all_embeddings)  # [batch_size, 576, 2048]
                batch_data['masks'] = np.stack(all_masks)
                
                try:
                    # 一次性保存整个batch（256个样本 → 1次I/O，超快！）
                    np.savez(batch_file, **batch_data)
                    saved_count = len(batch_names)
                    idx += saved_count
                    
                    if batch_idx % 5 == 0:
                        print(f"[Rank {process_id}] 批次 {batch_idx} 批量保存成功（2D VAE），文件: {os.path.basename(batch_file)}")
                except Exception as e:
                    print(f'[Rank {process_id}] 批量保存失败: {e}')
                    import traceback
                    traceback.print_exc()
            
            save_time = time.time() - save_start
            if batch_idx % 5 == 0:
                if save_time > 0.01:
                    speedup = 105.0 / save_time
                    print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} 文件保存耗时: {save_time:.2f}秒 (提速约{speedup:.1f}x)")
                else:
                    print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} 文件保存耗时: {save_time:.2f}秒 (极快！)")
            
            # 打印批次完成信息和总耗时
            batch_total_time = time.time() - batch_start_time
            if batch_idx % 5 == 0:
                print(f"[Rank {accelerator.process_index}] 批次 {batch_idx} 处理完成，总耗时: {batch_total_time:.2f}秒")
            
            # 如果某个批次耗时过长，打印警告
            if batch_total_time > 300:  # 5分钟
                print(f"[Rank {accelerator.process_index}] 警告：批次 {batch_idx} 耗时过长 ({batch_total_time:.2f}秒)")
            
            # 更新进度条
            if accelerator.is_main_process:
                progress_bar.update(1)
            
            # 每隔一定批次添加同步点，防止进程间差距过大
            # 使用中等同步频率平衡性能和安全性（每10批次同步一次）
            # 过于频繁会影响性能，过于稀疏可能导致hang检测延迟
            if (batch_idx + 1) % 10 == 0:
                # 先确保本进程的CUDA操作完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 每个进程都打印到达同步点的信息
                sync_time = time.time()
                print(f"[Rank {accelerator.process_index}] ⏱ [{time.strftime('%H:%M:%S')}] 到达同步点，批次 {batch_idx + 1}/{len(dataloader)}，已保存 {idx} 个文件")
                
                if accelerator.is_main_process:
                    print(f"\n[同步检查点] 已完成 {batch_idx + 1}/{len(dataloader)} 批次，所有进程准备同步...")
                
                # 添加同步超时保护
                try:
                    # 所有进程同步
                    accelerator.wait_for_everyone()
                    sync_duration = time.time() - sync_time
                    
                    if accelerator.is_main_process:
                        print(f"[同步完成] 所有进程已同步到批次 {batch_idx + 1}，同步耗时: {sync_duration:.2f}秒\n")
                    
                    # 同步后清理CUDA缓存，防止内存碎片
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as sync_error:
                    print(f"[Rank {accelerator.process_index}] 同步失败: {sync_error}")
                    raise
    
      except Exception as e:
          # 捕获异常，确保所有进程都能正确退出
          processing_error = True
          print(f"\n[Rank {accelerator.process_index}] 发生异常: {e}")
          import traceback
          traceback.print_exc()
          
          # 重要：即使发生异常，也要尝试通知其他进程
          # 避免其他进程永远等待在同步点
          try:
              # 等待其他进程（让它们也有机会检测到问题）
              print(f"[Rank {accelerator.process_index}] 尝试通知其他进程...")
              accelerator.wait_for_everyone()
          except Exception as sync_err:
              print(f"[Rank {accelerator.process_index}] 同步失败（预期行为）: {sync_err}")
      
      finally:
          # 关闭进度条
          if accelerator.is_main_process:
              progress_bar.close()
              print(f"\n[主进程] 所有批次处理完成，等待其他进程...")
    
    # 等待所有进程完成处理
    accelerator.wait_for_everyone()
    
    # 如果发生错误，提前退出
    if processing_error:
        if accelerator.is_main_process:
            print(f"[错误] 由于处理过程中出现错误，程序提前退出")
        return
    
    if accelerator.is_main_process:
        print(f"[主进程] 所有进程已完成，开始收集统计信息...")
    
    # 汇总所有进程的统计信息
    all_idx = accelerator.gather(torch.tensor([idx], device=device))
    
    # 收集失败样本列表
    all_failed_lists = gather_object([failed_samples])
    
    # 获取源目录的总图片数（在主进程外计算，所有进程都需要这个信息）
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_source_images = 0
    for fname in os.listdir(input_image_path):
        if os.path.splitext(fname)[1].lower() in valid_extensions:
            total_source_images += 1
    
    # 只在主进程打印统计信息和保存失败列表
    if accelerator.is_main_process:
        # 汇总成功处理的文件数
        total_saved = all_idx.sum().item()
        print(f'\n==================')
        print(f'处理完成！')
        print(f'成功保存: {total_saved} 个文件')
        print(f'各进程处理数量: {all_idx.cpu().tolist()}')
        
        # 合并所有进程的失败样本（去重）
        if all_failed_lists:
            all_failed = []
            for proc_failed_list in all_failed_lists:
                if proc_failed_list:
                    all_failed.extend(proc_failed_list)
            all_failed = list(set(all_failed))
            
            if all_failed:
                print(f'失败样本数: {len(all_failed)}')
                with open(os.path.join(save_dir, "failed_samples.txt"), 'w') as f:
                    for name in sorted(all_failed):
                        f.write(name + "\n")
                print(f'失败样本列表已保存到: {os.path.join(save_dir, "failed_samples.txt")}')
        
        # 保存详细的处理记录
        import datetime
        log_file = os.path.join(save_dir, "processing_log.txt")
        
        # 统计当前总处理量
        total_npy_files = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
        
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'='*80}\n")
            f.write(f"处理完成时间: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"本次处理:\n")
            f.write(f"  - 成功处理: {total_saved} 张图片\n")
            f.write(f"  - 各GPU处理数量: {all_idx.cpu().tolist()}\n")
            
            if all_failed_lists:
                all_failed = []
                for proc_failed_list in all_failed_lists:
                    if proc_failed_list:
                        all_failed.extend(proc_failed_list)
                all_failed = list(set(all_failed))
                if all_failed:
                    f.write(f"  - 失败样本数: {len(all_failed)}\n")
            
            f.write(f"\n总体进度:\n")
            f.write(f"  - 源目录图片总数: {total_source_images} 张\n")
            f.write(f"  - 已处理总数: {total_npy_files} 张\n")
            f.write(f"  - 待处理数量: {total_source_images - total_npy_files} 张\n")
            progress_pct = (total_npy_files / total_source_images * 100) if total_source_images > 0 else 0
            f.write(f"  - 完成进度: {progress_pct:.2f}%\n")
            f.write(f"\n配置信息:\n")
            f.write(f"  - GPU数量: {accelerator.num_processes}\n")
            f.write(f"  - 批次大小: {bz}\n")
            f.write(f"  - 输入图像目录（embeddings/masks）: {input_image_path}\n")
            f.write(f"  - 输出图像目录（moments）: {output_image_path}\n")
            f.write(f"  - 保存目录: {save_dir}\n")
            f.write(f"{'='*80}\n")
        
        print(f'\n处理记录已保存到: {log_file}')
        print(f'所有特征已保存到: {save_dir}')
        print(f'\n 总体进度统计:')
        print(f'  源目录总图片数: {total_source_images}')
        print(f'  已处理总数: {total_npy_files}')
        print(f'  待处理数量: {total_source_images - total_npy_files}')
        print(f'  完成进度: {progress_pct:.2f}%')
        print(f'==================\n')

if __name__ == '__main__':
    # ====================================================================================
    # 双路径特征提取说明：
    # 
    # 本脚本支持从两个不同的图像路径提取不同的特征：
    # - INPUT_IMAGE_PATH: 输入图像路径（用于提取embeddings和masks）
    # - OUTPUT_IMAGE_PATH: 输出图像路径（用于提取moments）
    # 
    # 增量处理说明：
    # - 每次运行会自动扫描源目录的所有图片（包括新增的）
    # - 自动检测并跳过已处理的图片（基于保存目录中的 .npy 文件）
    # - 只处理未处理的图片
    # - 支持中断后继续处理
    # - 适合数据不断增加的场景
    # 
    # 性能优化（针对8卡H100 80GB）：
    # - 已实现Janus批量处理（5-10x加速）
    # - 默认batch size=1024（充分利用80GB显存）
    # - 优化的DataLoader配置（num_workers=16, prefetch_factor=8）
    # 
    # 使用方式：
    #   bash run_multi_gpu_2D.sh
    # 
    # 参数配置：
    #   在run_multi_gpu_2D.sh中修改以下环境变量：
    #   - BATCH_SIZE: 批次大小（例如：2048用于H100，512用于较小GPU）
    #   - INPUT_IMAGE_PATH: 输入图像路径
    #   - OUTPUT_IMAGE_PATH: 输出图像路径
    #   - SAVE_DIR: 保存目录
    # ====================================================================================
    
    main()  # 所有参数从环境变量读取
