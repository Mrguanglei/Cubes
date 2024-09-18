import torch

def clear_cuda_cache():
    """
    释放未被张量引用的 GPU 缓存。
    调用这个函数可以帮助减少 GPU 内存的碎片化，从而避免 CUDA out of memory 错误。
    """
    # 检查CUDA是否可用，避免在没有CUDA支持的环境中调用
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理缓存
        print("CUDA cache cleared!")
    else:
        print("CUDA is not available. No cache to clear.")
