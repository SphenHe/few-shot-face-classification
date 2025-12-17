"""
清理和重置脚本
用于删除之前的处理结果和缓存
"""

import shutil
from pathlib import Path

def clean_results():
    """删除之前的识别结果"""
    results_folder = Path("data/results")
    if results_folder.exists():
        print(f"删除结果文件夹: {results_folder}")
        shutil.rmtree(results_folder)
        print("✅ 已删除")
    else:
        print("✓ 结果文件夹不存在（已是干净状态）")

def clean_cache():
    """删除模型缓存和临时文件"""
    cache_folders = [
        Path.home() / ".cache" / "torch",  # PyTorch 模型缓存
        Path.home() / ".facenet_pytorch_data",  # facenet-pytorch 缓存
    ]

    # 删除本地嵌入缓存
    embeddings_cache = Path("data/embeddings_cache.pkl")
    if embeddings_cache.exists():
        print(f"删除嵌入缓存: {embeddings_cache}")
        try:
            embeddings_cache.unlink()
            print("✅ 已删除")
        except Exception as e:
            print(f"⚠️  删除失败: {e}")
    else:
        print(f"✓ 嵌入缓存不存在: {embeddings_cache}")

    for cache_dir in cache_folders:
        if cache_dir.exists():
            print(f"删除缓存: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print("✅ 已删除")
            except Exception as e:
                print(f"⚠️  删除失败: {e}")
        else:
            print(f"✓ 缓存目录不存在: {cache_dir}")

def reset_all():
    """完全重置 - 删除所有结果和缓存"""
    print("="*60)
    print("完全重置 - 清理所有结果和缓存")
    print("="*60)
    
    print("\n1️⃣  清理识别结果...")
    clean_results()
    
    print("\n2️⃣  清理模型缓存...")
    clean_cache()
    
    print("\n" + "="*60)
    print("✅ 重置完成！")
    print("="*60)
    print("\n现在可以重新运行分类：")
    print("  python run_classification.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            reset_all()
        elif sys.argv[1] == "--results":
            print("删除识别结果...")
            clean_results()
        elif sys.argv[1] == "--cache":
            print("删除模型缓存...")
            clean_cache()
        else:
            print("用法:")
            print("  python clean.py --results  删除识别结果")
            print("  python clean.py --cache    删除模型缓存")
            print("  python clean.py --all      删除所有结果和缓存")
    else:
        # 默认只删除结果
        print("删除识别结果...")
        clean_results()
        print("\n💡 提示：使用 'python clean.py --all' 删除所有结果和缓存")
