"""
人脸分类快速运行脚本
使用你的数据进行人脸识别和分类
"""

from pathlib import Path
from few_shot_face_classification import detect_and_export

def main():
    print("="*60)
    print("开始人脸识别和分类任务")
    print("="*60)
    
    # 定义数据路径
    DATA_RAW = Path("data/raw")          # 原始图片文件夹
    DATA_LABELED = Path("data/labeled")  # 标注人脸文件夹
    DATA_RESULTS = Path("data/results")  # 结果输出文件夹
    
    # 创建结果文件夹
    DATA_RESULTS.mkdir(exist_ok=True, parents=True)
    
    # 显示数据统计
    raw_images = list(DATA_RAW.glob("*.*"))
    labeled_images = list(DATA_LABELED.glob("*.*"))
    
    print(f"\n📁 数据统计:")
    print(f"  - 原始图片数量: {len(raw_images)}")
    print(f"  - 标注人脸数量: {len(labeled_images)}")
    
    # 统计标注的人数
    names = set()
    for img in labeled_images:
        name = img.stem.split('_')[0]  # 获取姓名部分
        if name != 'none':
            names.add(name)
    
    print(f"  - 目标识别人数: {len(names)}")
    print(f"\n🎯 目标人物列表:")
    for i, name in enumerate(sorted(names), 1):
        print(f"  {i:2d}. {name}")
    
    print(f"\n⏳ 开始处理...")
    print(f"  ⚠️  首次运行会下载预训练模型（约100MB），请耐心等待")
    print(f"  ⚠️  处理{len(raw_images)}张图片可能需要几分钟时间")
    print()
    
    # 执行分类
    try:
        detect_and_export(
            raw_f=DATA_RAW,
            labeled_f=DATA_LABELED,
            write_f=DATA_RESULTS,
            draw_boxes=True,  # 在输出照片上绘制人脸框框和名字
        )
        
        print("\n" + "="*60)
        print("✅ 分类完成！")
        print("="*60)
        print(f"\n📂 结果保存位置: {DATA_RESULTS.absolute()}")
        print(f"\n每个人的照片已保存在各自的子文件夹中：")
        
        # 显示结果统计
        result_folders = [f for f in DATA_RESULTS.iterdir() if f.is_dir()]
        for folder in sorted(result_folders):
            images_count = len(list(folder.glob("*.*")))
            if images_count > 0:
                print(f"  - {folder.name}: {images_count} 张图片")
        
        print("\n🎉 任务完成！现在可以查看结果文件夹。")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误：")
        print(f"  {str(e)}")
        print(f"\n请检查：")
        print(f"  1. 所有依赖包是否已正确安装")
        print(f"  2. 数据文件夹路径是否正确")
        print(f"  3. 标注图片格式是否符合要求（姓名_编号.格式）")
        raise

if __name__ == "__main__":
    main()
