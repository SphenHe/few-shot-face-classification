# Few-Shot Face Classification 使用指南

## 📋 项目概述

这是一个人脸识别和分类库，可以从大量图片中自动提取包含特定人物的照片。

**你的数据情况：**
- ✅ **标注数据** (`/data/labeled/`): 310张已标注的人脸图片，包含31个人（每人10张）
- ✅ **原始数据** (`/data/raw/`): 287张待分类的原始图片
- 🎯 **目标**: 从原始图片中识别出包含这31个人的照片

---

## 🚀 环境设置步骤

### 1. 安装 Python 环境
确保你已安装 Python 3.7 或更高版本。

### 2. 安装项目依赖
在项目根目录下运行以下命令：

```powershell
# 方式1: 使用 pip 直接安装（推荐）
pip install -e .

# 方式2: 手动安装所需包
pip install facenet-pytorch Pillow matplotlib scikit-learn tqdm numpy
```

### 3. 验证安装
```powershell
python -c "from few_shot_face_classification import detect_and_export; print('安装成功！')"
```

---

## 📁 数据准备说明

### 你的数据已经正确组织：

#### 1. **标注数据** (`/data/labeled/`)
- ✅ 格式正确：`<姓名>_<编号>.<格式>`
- ✅ 例如：`AAA_1.jpg`, `AAA_2.jpg`, ...
- ✅ 每张图片只包含一个人脸
- ✅ 共识别到31个人：
  - AAA, BBB, CCC, DDD, EEE, FFF, GGG, HHH
  - III, JJJ, KKK, LLL, MMM, NNN, OOO, PPP
  - QQQ, ...

#### 2. **原始数据** (`/data/raw/`)
- ✅ ???张待分类图片
- ✅ 图片可以包含多个人或无人脸

---

## 🎯 使用方法

### 方法1: 使用核心函数（最简单）

创建一个 Python 脚本 `run_classification.py`:

```python
from pathlib import Path
from few_shot_face_classification import detect_and_export

# 定义数据路径
DATA_RAW = Path("data/raw")          # 原始图片文件夹
DATA_LABELED = Path("data/labeled")  # 标注人脸文件夹
DATA_RESULTS = Path("data/results")  # 结果输出文件夹

# 执行分类
detect_and_export(
    raw_f=DATA_RAW,
    labeled_f=DATA_LABELED,
    write_f=DATA_RESULTS,
)

print("✅ 分类完成！结果保存在 data/results/ 文件夹中")
print("每个人的照片会保存在对应的子文件夹中，例如：")
print("  - data/results/AAA/")
print("  - data/results/BBB/")
print("  等等...")
```

然后运行：
```powershell
python run_classification.py
```

### 方法2: 识别单张图片中的人物

```python
from pathlib import Path
from few_shot_face_classification import recognise

# 识别单张图片
im_path = Path("data/raw/微信图片_20251207094244_176_90.jpg")
classes = recognise(
    path=im_path,
    labeled_f=Path("data/labeled"),
)

print(f"识别到的人物: {classes}")
```

### 方法3: 使用 Jupyter Notebook（交互式）

已为你准备好 `demo/main.ipynb`，你可以：
1. 打开 Jupyter Notebook
2. 修改数据路径为你的 `/data/` 文件夹
3. 逐步运行查看效果

---

## 📊 工作原理

1. **加载标注数据**: 从 `labeled/` 文件夹学习每个人的人脸特征
2. **人脸检测**: 使用 MTCNN 网络检测图片中的人脸
3. **特征提取**: 使用 VGGFace2 网络提取人脸特征（embeddings）
4. **相似度匹配**: 计算欧氏距离，距离 < 1.0 认为是同一个人
5. **导出结果**: 将匹配到的图片复制到对应人物的文件夹

---

## ⚙️ 高级功能

### 添加"非目标人物"
如果有很多误识别，可以添加非目标人物的照片：

```python
from few_shot_face_classification import add_none

# 将某张图片中的人脸标记为"非目标人物"
add_none(
    path=Path("data/raw/某张照片.jpg"),
    labeled_f=Path("data/labeled"),
)
```

这会在 `labeled/` 文件夹中生成 `none_<编号>.jpg`，这些人脸将被忽略。

### 验证标注数据质量

```python
from few_shot_face_classification import validate_labels

# 验证所有标注数据是否合格
validate_labels(
    labeled_f=Path("data/labeled"),
    handle="warn"  # 可选: "ignore", "warn", "raise"
)
```

---

## 📝 常见问题

### Q1: 某些人没有被识别出来？
**A**: 尝试在 `labeled/` 文件夹中添加更多该人的不同角度照片（建议5-10张）。

### Q2: 有很多误识别？
**A**: 使用 `add_none()` 函数将误识别的人脸添加为"非目标人物"。

### Q3: 处理速度慢？
**A**: 第一次运行会下载预训练模型（约100MB），之后会快很多。

### Q4: 一张照片出现在多个人的文件夹中？
**A**: 这是正常的！如果一张照片包含多个目标人物，会被复制到多个文件夹。

---

## 🎨 结果查看

运行完成后，查看 `data/results/` 文件夹：

```
data/results/
├── AAA/
│   ├── 图片1.jpg
│   ├── 图片2.jpg
│   └── ...
├── BBB/
│   ├── 图片1.jpg
│   └── ...
└── ... (其他??个人的文件夹)
```

---

## 💡 推荐工作流程

1. **首次运行**: 先运行小样本测试（例如只用部分 raw 图片）
2. **检查结果**: 查看识别准确率
3. **调整优化**: 
   - 如果某人识别率低 → 添加更多该人的标注照片
   - 如果误识别多 → 使用 `add_none()` 排除
4. **完整运行**: 对所有图片执行分类

---

## 📞 需要帮助？

- 查看 `demo/main.ipynb` 获取更详细的示例
- 阅读项目 README.md
- 遇到问题可以查看错误信息并相应调整

祝使用顺利！🎉
