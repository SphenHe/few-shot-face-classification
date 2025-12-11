"""
增强的图像可视化工具
为图像上的人脸添加框框和姓名标签
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，解决中文显示问题

from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

from few_shot_face_classification.embed import get_networks
from few_shot_face_classification.data import load_single, get_im_paths
from few_shot_face_classification.similarity import get_classes
from few_shot_face_classification.embed import embed as embed_func


def visualize_faces_with_boxes(
    image_path: Path,
    labeled_f: Path,
    names_to_highlight: Optional[List[str]] = None,
    thr: float = 1.0,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    可视化图像中的人脸，并用框框和名字标注出来
    
    :param image_path: 图像路径
    :param labeled_f: 标注人脸文件夹路径
    :param names_to_highlight: 需要高亮显示的人名列表（如果为None则显示所有识别到的人）
    :param thr: 识别阈值
    :param figsize: 图表大小
    """
    from few_shot_face_classification.embed import embed
    
    # 加载图像
    im = load_single(image_path)
    
    # 获取人脸检测网络
    mtcnn, vggface2 = get_networks()
    
    # 检测人脸并获取坐标
    batch_boxes, _ = mtcnn.detect(im)
    
    if batch_boxes is None:
        print("未检测到人脸")
        return
    
    # 获取人脸特征
    embs = embed(im, mtcnn=mtcnn, vggface2=vggface2)
    
    # 加载标注数据
    labeled_paths, labeled_embs = [], []
    labels_map = {}  # 人脸索引 -> 人名
    
    for labeled_path in sorted(get_im_paths(labeled_f)):
        # 提取人名
        name = labeled_path.stem.split('_')[0]
        if name == 'none':
            continue
            
        # 加载特征
        im_labeled = load_single(labeled_path)
        try:
            emb_labeled = embed_func(im_labeled, mtcnn=mtcnn, vggface2=vggface2)
            if emb_labeled:
                labeled_paths.append(labeled_path)
                labeled_embs.append(emb_labeled[0])
                labels_map[len(labeled_embs) - 1] = name
        except Exception:
            continue
    
    if not labeled_embs:
        print("没有有效的标注数据")
        return
    
    labeled_embs = np.array(labeled_embs)
    
    # 获取识别结果
    classes = get_classes(
        embs=embs,
        labeled_paths=labeled_paths,
        labeled_embs=labeled_embs,
        thr=thr,
    )
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 显示原始图像
    im_array = np.array(im)
    ax.imshow(im_array)
    
    # 在检测到的人脸上绘制框框和标签
    for i, (box, name) in enumerate(zip(batch_boxes, classes)):
        if name is None or (names_to_highlight and name not in names_to_highlight):
            color = 'red'  # 未识别或非目标人物用红色
            label = f"Unknown #{i+1}"
        else:
            color = 'green'  # 识别到的目标人物用绿色
            label = name
        
        # 绘制框框
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加文字标签
        ax.text(
            x1, y1 - 5,
            label,
            fontsize=12,
            color='black',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
            ha='left'
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_class_folder(
    class_folder: Path,
    cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    class_name: Optional[str] = None
) -> None:
    """
    可视化一个人物文件夹中的所有照片
    
    :param class_folder: 人物文件夹路径
    :param cols: 每行显示的列数
    :param figsize: 图表大小（如果为None则自动计算）
    :param class_name: 人物名字（如果为None则从文件夹名推断）
    """
    # 获取人物名字
    if class_name is None:
        class_name = class_folder.name
    
    # 获取所有图像
    im_paths = get_im_paths(class_folder)
    
    if not im_paths:
        print(f"文件夹 {class_folder} 中没有图像")
        return
    
    # 计算行数
    rows = (len(im_paths) + cols - 1) // cols
    
    # 自动计算图表大小
    if figsize is None:
        figsize = (cols * 4, rows * 4)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"'{class_name}' 的识别结果 ({len(im_paths)} 张)", fontsize=16, fontweight='bold')
    
    for i, im_path in enumerate(im_paths, 1):
        ax = fig.add_subplot(rows, cols, i)
        
        # 读取和显示图像
        im = plt.imread(im_path)
        ax.imshow(im)
        ax.set_title(im_path.name, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_all_classes(
    results_folder: Path,
    max_images_per_class: int = 5,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    可视化所有识别出来的人物及其代表图像
    
    :param results_folder: 结果文件夹路径
    :param max_images_per_class: 每个人显示的最大图像数
    :param figsize: 图表大小
    """
    # 获取所有人物文件夹
    class_folders = sorted([f for f in results_folder.iterdir() if f.is_dir()])
    
    if not class_folders:
        print(f"结果文件夹 {results_folder} 中没有人物文件夹")
        return
    
    print(f"识别到 {len(class_folders)} 个人物")
    
    # 对每个人物创建一个小图表
    for class_folder in class_folders:
        im_paths = get_im_paths(class_folder)
        
        if not im_paths:
            continue
        
        # 限制显示的图像数量
        im_paths = im_paths[:max_images_per_class]
        
        # 创建小图表
        cols = min(len(im_paths), 5)
        rows = (len(im_paths) + cols - 1) // cols
        
        if figsize is None:
            fig_size = (cols * 3, rows * 3)
        else:
            fig_size = figsize
        
        fig = plt.figure(figsize=fig_size)
        fig.suptitle(
            f"{class_folder.name} ({len(get_im_paths(class_folder))} 张)",
            fontsize=14,
            fontweight='bold'
        )
        
        for i, im_path in enumerate(im_paths, 1):
            ax = fig.add_subplot(rows, cols, i)
            im = plt.imread(im_path)
            ax.imshow(im)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 使用示例
    from few_shot_face_classification.data import get_im_paths
    from pathlib import Path
    
    # 配置路径
    DATA_RAW = Path("demo/raw")
    DATA_LABELED = Path("demo/labeled")
    DATA_RESULTS = Path("demo/results")
    
    # 示例1: 显示单张图像的人脸识别结果（带框框和名字）
    print("示例1: 显示单张图像的人脸识别")
    raw_images = get_im_paths(DATA_RAW)
    if raw_images:
        visualize_faces_with_boxes(raw_images[0], DATA_LABELED)
    
    # 示例2: 显示某个人物的所有识别照片
    print("\n示例2: 显示某个人物的所有照片")
    if (DATA_RESULTS / "sheldon").exists():
        visualize_class_folder(DATA_RESULTS / "sheldon", class_name="Sheldon")
    
    # 示例3: 显示所有识别结果
    print("\n示例3: 显示所有人物的识别结果")
    visualize_all_classes(DATA_RESULTS)
