# Stage C (SSDA) 数据集下载与配置指南

本文档说明如何从零准备 Stage C（Semi-Supervised Domain Adaptation）所需的数据集，以及如何跑通 GTA→Cityscapes 实验。

---

## 一、Stage C 需要的数据集

### 1.1 必须数据（GTA→Cityscapes）

| 数据集 | 角色 | 用途 | 是否已有 |
|--------|------|------|----------|
| **Cityscapes** | Target | 目标域图像 + 部分 labeled mask（100/500） | 你已有（Stage A/B） |
| **GTA5 Segmentation** | Source | 源域图像 + 全量语义标签 | 需下载 |

### 1.2 可选数据（如需 SYNTHIA→Cityscapes 实验）

| 数据集 | 用途 | 是否需要 |
|--------|------|----------|
| SYNTHIA-RAND-CITYSCAPES | SYNTHIA→Cityscapes 实验 | 若做 SYNTHIA 源域则需要 |
| SYNTHIA→Cityscapes 映射标签 | SYNTHIA 标签映射到 19 类 | 同上，**必须** |
| GTA5 Video Sequences | 在 GTA 上预训练 SDE | 可选（可用论文预训练） |

---

## 二、数据下载步骤

### 2.1 Cityscapes（已有则跳过）

你已经用于 Stage A/B，保持现有结构即可。

期望目录结构（与 machine_config 路径对应）：

```
/scratch/u5hv/shijie.u5hv/sde_seg/raw/   # 或 MachineConfig.CITYSCAPES_DIR
├── gtFine/
├── leftImg8bit_trainvaltest/
└── leftImg8bit_sequence/   # 或 leftImg8bit_sequence_small/
```

### 2.2 GTA5 分割数据集（必须）

**下载来源**：  
https://download.visinf.tu-darmstadt.de/data/from_games/

**所需文件**（共 20 个 zip）：
- `01_labels.zip` ~ `10_labels.zip`（标签）
- `01_images.zip` ~ `10_images.zip`（图像）

**方式一：使用官方脚本**

```bash
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

# 切换到 ssda 分支以获取下载脚本
git fetch upstream
git checkout ssda  # 或 git checkout upstream/ssda -b ssda

# 创建下载目录并进入
mkdir -p datasets/GTASeg
cd datasets/GTASeg

# 执行下载（需提前 checkout ssda 分支）
bash ../../data_preprocessing/gta_seg_download.sh

# 解压后整理目录结构，应得到：
# datasets/GTASeg/
# ├── images/     # 01/, 02/, ..., 10/ 子目录
# └── labels/     # 01/, 02/, ..., 10/ 子目录
```

**方式二：手动 wget**

```bash
mkdir -p /scratch/u5hv/shijie.u5hv/sde_seg/datasets/GTASeg
cd /scratch/u5hv/shijie.u5hv/sde_seg/datasets/GTASeg

# 下载 labels
for i in $(seq -w 1 10); do
  wget --no-check-certificate https://download.visinf.tu-darmstadt.de/data/from_games/data/${i}_labels.zip
done

# 下载 images
for i in $(seq -w 1 10); do
  wget --no-check-certificate https://download.visinf.tu-darmstadt.de/data/from_games/data/${i}_images.zip
done

# 解压
unzip -n '*.zip'
```

解压后 GTA5 的原始结构通常是 `images/` 和 `labels/` 两个顶层目录，loader 期望的就是这种结构。

**GTA5 Label Mapping**：  
已在 `loader/gta_seg_loader.py` 中硬编码，GTA 原始 ID 映射到 Cityscapes 19 类，无需额外 mapping 文件。

### 2.3 SYNTHIA-RAND-CITYSCAPES（SYNTHIA→Cityscapes 实验需要）

若做 SYNTHIA→Cityscapes，需同时准备：

1. **SYNTHIA-RAND-CITYSCAPES 原始数据**：  
   http://synthia-dataset.net/download/808/  
   下载后解压到 `RGB/` 子目录。

2. **映射到 Cityscapes 的标签**（**必须**，loader 依赖此格式）：  
   https://drive.google.com/file/d/1cwGJGzE8yEvKTzF2KeaBcydnVZmng-KL/view?usp=sharing  
   下载后解压到 `synthia_mapped_to_cityscapes/` 子目录。

3. **目录结构**（放在 `/scratch/u5hv/shijie.u5hv/sde_seg/datasets/Synthia/`）：
   ```
   Synthia/
   ├── RGB/                           # 原始 RGB 图像（或 RGB_small/）
   └── synthia_mapped_to_cityscapes/  # 映射后的标签（必须）
   ```

4. **Google Drive 下载**：可用 `gdown` 或浏览器手动下载：
   ```bash
   pip install gdown
   gdown "https://drive.google.com/uc?id=1cwGJGzE8yEvKTzF2KeaBcydnVZmng-KL" -O synthia_mapped_to_cityscapes.zip
   unzip synthia_mapped_to_cityscapes.zip -d /scratch/u5hv/shijie.u5hv/sde_seg/datasets/Synthia/
   ```

---

## 三、machine_config 与路径配置

ssda 分支的 `configs/machine_config.py` 使用相对路径（`datasets/...`），你的 Stage A/B 使用绝对路径（`/scratch/...`）。需要统一。

**方案 A：使用相对路径（在 repo 根目录运行）**

保持 ssda 默认配置，在 repo 根目录创建软链接：

```bash
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth
mkdir -p datasets
ln -sf /scratch/u5hv/shijie.u5hv/sde_seg/raw datasets/Cityscapes
ln -sf /scratch/u5hv/shijie.u5hv/sde_seg/datasets/GTASeg datasets/GTASeg
```

**方案 B：修改 machine_config 使用绝对路径（推荐）**

在 ssda 分支的 `configs/machine_config.py` 中，为 `ws` 机器添加：

```python
if machine == "ws":
    MachineConfig.CITYSCAPES_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/raw/"
    MachineConfig.GTASEQ_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/datasets/GTASeq/"
    MachineConfig.GTASEG_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/datasets/GTASeg/"
    MachineConfig.SYNTHIA_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/datasets/Synthia/"
    # ... 其余保持不变
```

---

## 四、下采样（可选但推荐）

为加速训练，可对图像做下采样。ssda 分支提供：

```bash
python -m data_preprocessing.downsample_datasets --machine ws
```

会生成：
- `Cityscapes/leftImg8bit_small/`
- `Cityscapes/leftImg8bit_sequence_small/`
- `GTASeg/images_small/`
- `Synthia/RGB_small/`、`Synthia/video_small/`

loader 会根据 `img_size` 自动选择 `images` / `images_small`。若不做下采样，保持原始分辨率亦可。

---

## 五、快速检查清单

运行 Stage C 前，确认：

- [ ] 已切换到 `ssda` 分支
- [ ] GTA5 已下载并解压到 `GTASeg/images/` 和 `GTASeg/labels/`
- [ ] Cityscapes 路径正确（gtFine, leftImg8bit_trainvaltest）
- [ ] （若做 SYNTHIA）SYNTHIA 已下载，含 `RGB/` 和 `synthia_mapped_to_cityscapes/`
- [ ] `machine_config.py` 中 `ws` 的路径指向实际数据位置
- [ ] （可选）已运行 `downsample_datasets` 生成 `_small` 版本

---

## 六、最小可运行测试

```bash
git checkout ssda   # 或 upstream/ssda -b ssda
python run_experiments.py --machine ws --config configs/ssda.yml --exp 260
```

exp 260 会跑 SSDA baseline（无 MTL）。若 dataloader 正常，日志中应能看到 source (GTA) 与 target (Cityscapes) 的加载信息。

---

## 七、下载链接汇总

| 数据集 | 链接 |
|--------|------|
| GTA5 Images/Labels | https://download.visinf.tu-darmstadt.de/data/from_games/ |
| SYNTHIA-RAND-CITYSCAPES | http://synthia-dataset.net/download/808/ |
| SYNTHIA→Cityscapes 映射标签 | https://drive.google.com/file/d/1cwGJGzE8yEvKTzF2KeaBcydnVZmng-KL/view?usp=sharing |
| GTA5 Video（可选） | https://playing-for-benchmarks.org/download/ |
