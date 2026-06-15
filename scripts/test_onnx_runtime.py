"""
ONNX Runtime GPU 性能测试脚本
测试 YOLO11m-pose 模型在不同后端下的推理性能
"""
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
import cv2
import glob

# 配置
ONNX_MODEL_PATH = Path("models/yolo/yolo11m-pose.onnx")
TEST_IMAGE_DIR = Path("test")
TEST_DEBUG_DIR = Path("data/debug")

def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("ONNX Runtime GPU 环境检查")
    print("=" * 60)
    
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
    has_tensorrt = 'TensorrtExecutionProvider' in ort.get_available_providers()
    
    print(f"CUDA Execution Provider: {'Available' if has_cuda else 'Not Available'}")
    print(f"TensorRT Execution Provider: {'Available' if has_tensorrt else 'Not Available'}")
    
    return has_cuda, has_tensorrt

def load_model(use_gpu=True):
    """加载 ONNX 模型"""
    if not ONNX_MODEL_PATH.exists():
        print(f"Model not found: {ONNX_MODEL_PATH}")
        return None
    
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    print(f"Loading model with providers: {providers}")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
    
    return session

def preprocess_image(image_path, input_size=640):
    """预处理图像"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # 保持原始尺寸比例缩放
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建画布并居中放置
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    x_offset = (input_size - new_w) // 2
    y_offset = (input_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 转换为模型输入格式 (1, 3, 640, 640)
    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    
    return blob, image, scale, x_offset, y_offset

def run_inference(session, blob):
    """运行推理"""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return outputs

def postprocess_pose(outputs, conf_threshold=0.5):
    """后处理姿态估计结果"""
    # outputs[0] shape: (1, 56, 8400)
    # 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3)
    predictions = outputs[0][0]  # (56, 8400)
    
    # 转置为 (8400, 56)
    predictions = predictions.T
    
    results = []
    for pred in predictions:
        # 解析边界框
        cx, cy, w, h = pred[:4]
        conf = pred[4]
        
        if conf < conf_threshold:
            continue
        
        # 解析关键点 (17 * 3 = 51)
        keypoints = []
        for i in range(17):
            kx = pred[5 + i * 3]
            ky = pred[5 + i * 3 + 1]
            kconf = pred[5 + i * 3 + 2]
            keypoints.append((kx, ky, kconf))
        
        results.append({
            'bbox': (cx, cy, w, h),
            'confidence': conf,
            'keypoints': keypoints
        })
    
    return results

def benchmark_inference(session, num_warmup=5, num_iterations=20):
    """性能基准测试"""
    # 创建随机输入
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # 预热
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        run_inference(session, dummy_input)
    
    # 基准测试
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        run_inference(session, dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'times': times
    }

def test_with_real_images(session, image_dir, max_images=5):
    """使用真实图像测试"""
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return []
    
    # 查找图像文件
    image_files = []
    for ext in ['*.jpg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(str(image_dir / ext)))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    results = []
    for img_path in image_files[:max_images]:
        print(f"\nTesting: {Path(img_path).name}")
        
        # 预处理
        preprocessed = preprocess_image(img_path)
        if preprocessed is None:
            print(f"  Failed to load image")
            continue
        
        blob, original_image, scale, x_offset, y_offset = preprocessed
        
        # 推理
        start = time.perf_counter()
        outputs = run_inference(session, blob)
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000
        
        # 后处理
        pose_results = postprocess_pose(outputs, conf_threshold=0.5)
        
        print(f"  Inference time: {inference_time:.1f}ms")
        print(f"  Detected persons: {len(pose_results)}")
        
        if pose_results:
            for i, pose in enumerate(pose_results):
                print(f"  Person {i+1}: conf={pose['confidence']:.3f}, "
                      f"bbox=({pose['bbox'][0]:.0f}, {pose['bbox'][1]:.0f}, "
                      f"{pose['bbox'][2]:.0f}, {pose['bbox'][3]:.0f})")
        
        results.append({
            'image': img_path,
            'time': inference_time,
            'persons': len(pose_results),
            'poses': pose_results
        })
    
    return results

def main():
    """主测试函数"""
    print("=" * 60)
    print("ONNX Runtime GPU 性能测试")
    print("=" * 60)
    
    # 检查环境
    has_cuda, has_tensorrt = check_environment()
    
    if not has_cuda:
        print("CUDA is not available, testing CPU only")
    
    # 测试 CPU 推理
    print("\n" + "=" * 60)
    print("CPU 推理测试")
    print("=" * 60)
    
    session_cpu = load_model(use_gpu=False)
    if session_cpu:
        cpu_stats = benchmark_inference(session_cpu)
        print(f"\nCPU Performance:")
        print(f"  Mean: {cpu_stats['mean']:.1f}ms")
        print(f"  Std: {cpu_stats['std']:.1f}ms")
        print(f"  Min: {cpu_stats['min']:.1f}ms")
        print(f"  Max: {cpu_stats['max']:.1f}ms")
        print(f"  Median: {cpu_stats['median']:.1f}ms")
        
        # 测试真实图像
        print("\n" + "-" * 40)
        print("CPU 真实图像测试")
        print("-" * 40)
        
        # 测试 debug 目录
        if TEST_DEBUG_DIR.exists():
            print(f"\nTesting debug directory: {TEST_DEBUG_DIR}")
            cpu_debug_results = test_with_real_images(session_cpu, TEST_DEBUG_DIR)
        
        # 测试 test 目录
        if TEST_IMAGE_DIR.exists():
            print(f"\nTesting test directory: {TEST_IMAGE_DIR}")
            cpu_test_results = test_with_real_images(session_cpu, TEST_IMAGE_DIR)
    
    # 测试 GPU 推理
    if has_cuda:
        print("\n" + "=" * 60)
        print("GPU (CUDA) 推理测试")
        print("=" * 60)
        
        session_gpu = load_model(use_gpu=True)
        if session_gpu:
            gpu_stats = benchmark_inference(session_gpu)
            print(f"\nGPU Performance:")
            print(f"  Mean: {gpu_stats['mean']:.1f}ms")
            print(f"  Std: {gpu_stats['std']:.1f}ms")
            print(f"  Min: {gpu_stats['min']:.1f}ms")
            print(f"  Max: {gpu_stats['max']:.1f}ms")
            print(f"  Median: {gpu_stats['median']:.1f}ms")
            
            # 计算加速比
            if session_cpu:
                speedup = cpu_stats['mean'] / gpu_stats['mean']
                print(f"\nSpeedup: {speedup:.2f}x")
            
            # 测试真实图像
            print("\n" + "-" * 40)
            print("GPU 真实图像测试")
            print("-" * 40)
            
            # 测试 debug 目录
            if TEST_DEBUG_DIR.exists():
                print(f"\nTesting debug directory: {TEST_DEBUG_DIR}")
                gpu_debug_results = test_with_real_images(session_gpu, TEST_DEBUG_DIR)
            
            # 测试 test 目录
            if TEST_IMAGE_DIR.exists():
                print(f"\nTesting test directory: {TEST_IMAGE_DIR}")
                gpu_test_results = test_with_real_images(session_gpu, TEST_IMAGE_DIR)
    
    # 测试 TensorRT 推理
    if has_tensorrt:
        print("\n" + "=" * 60)
        print("GPU (TensorRT) 推理测试")
        print("=" * 60)
        
        try:
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
            session_trt = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
            
            trt_stats = benchmark_inference(session_trt)
            print(f"\nTensorRT Performance:")
            print(f"  Mean: {trt_stats['mean']:.1f}ms")
            print(f"  Std: {trt_stats['std']:.1f}ms")
            print(f"  Min: {trt_stats['min']:.1f}ms")
            print(f"  Max: {trt_stats['max']:.1f}ms")
            print(f"  Median: {trt_stats['median']:.1f}ms")
            
            # 计算加速比
            if session_cpu:
                speedup = cpu_stats['mean'] / trt_stats['mean']
                print(f"\nSpeedup vs CPU: {speedup:.2f}x")
            
        except Exception as e:
            print(f"TensorRT test failed: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
