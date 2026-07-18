# Few-Shot Face Classification

从 `data/raw/` 中识别包含目标人物的照片，并把结果导出到 `data/results/`。

## 环境配置

推荐一键配置，脚本会优先使用 Conda；如果当前环境没有 `conda`，才会使用 Python venv。

```bash
python3 setup_env.py
```

如果使用 Conda，环境创建后激活：

```bash
conda activate few-shot-face-classification
```

也可以直接使用 Conda 配置文件：

```bash
conda env create -f environment.yml
conda activate few-shot-face-classification
```

强制使用 Python venv：

```bash
python3 setup_env.py --env-manager venv
source .venv/bin/activate
```

Windows PowerShell 激活 venv：

```powershell
.\.venv\Scripts\Activate.ps1
```

已有环境时直接安装：

```bash
python -m pip install -e .
```

依赖环境需要 Python 3.8 到 3.12。Python 3.13 不推荐，当前人脸识别依赖栈可能触发源码编译失败。

验证安装：

```bash
python -c "from few_shot_face_classification import detect_and_export; print('OK')"
```

## 数据目录

所有数据统一放在 `data/`：

```text
data/
├── labeled/
├── raw/
├── results/              # 脚本生成
└── embeddings_cache.pkl  # 脚本生成
```

`data/labeled/` 中每张图片应只包含一个人脸，文件名格式为 `姓名_编号.jpg`，例如：

```text
data/labeled/
├── sheldon_1.jpg
├── sheldon_2.jpg
├── penny_1.jpg
└── none_1.jpg
```

`none_*.jpg` 表示非目标人物，可用于减少误识别。

`data/raw/` 中放待分类图片，可以包含多人、单人或无人脸。

## 运行

```bash
python run_classification.py
```

结果会写入 `data/results/`，每个识别到的人物一个子目录。

摄像头测试：

```bash
python test-cv.py
```

实时识别：

```bash
python video_realtime.py
```

清理结果：

```bash
python clean.py --results
```

清理结果和缓存：

```bash
python clean.py --all
```

## Python API

批量识别并导出：

```python
from pathlib import Path
from few_shot_face_classification import detect_and_export

detect_and_export(
    raw_f=Path("data/raw"),
    labeled_f=Path("data/labeled"),
    write_f=Path("data/results"),
)
```

默认 `conflict=Conflict.MOVE`，标注图片无效时会移动到 `data/error_data/` 后继续处理；目录不存在时会自动创建，重名时会自动保留两份文件。
如果希望遇到无效标注图片时直接停止且不移动或删除 `data/labeled/` 中的文件，可显式传入 `conflict=Conflict.CRASH`。

识别单张图片：

```python
from pathlib import Path
from few_shot_face_classification import recognise

classes = recognise(
    path=Path("data/raw/example.jpg"),
    labeled_f=Path("data/labeled"),
)
print(classes)
```

添加非目标人物：

```python
from pathlib import Path
from few_shot_face_classification import add_none

add_none(
    path=Path("data/raw/example.jpg"),
    labeled_f=Path("data/labeled"),
)
```

## 常见问题

某些人没有被识别出来：给该人物在 `data/labeled/` 中增加更多不同角度、光照、表情的单人脸照片。

误识别较多：把容易被误识别的人脸加入 `none_*.jpg`，或使用 `add_none()` 从图片中提取非目标人脸。

第一次运行较慢：首次运行会下载预训练模型，之后会使用缓存。
