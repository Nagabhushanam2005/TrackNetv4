import torch
import time
import numpy as np
from TrackNetV4_CSPNeXt import TrackNetV4_CSPNeXt_BallTracking
from TrackNetv4_EfficientNet import TrackNetV4_EfficientUNet

# Import your original VGG-based model
# Assuming it's available as TrackNetV4
try:
    from TrackNetV4_pt import TrackNetV4  # Adjust import as needed
    HAS_ORIGINAL = True
except ImportError:
    print("Warning: Original TrackNetV4 not found. Will only benchmark CSPNeXt model.")
    HAS_ORIGINAL = False


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model(model, input_shape, num_warmup=10, num_iterations=100, device='cuda'):
    """Benchmark model inference speed."""
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Running benchmark ({num_iterations} iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }


def print_benchmark_results(name, stats, params):
    """Print benchmark results in a formatted way."""
    total_params, trainable_params = params
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  Total:      {total_params:>15,}")
    print(f"  Trainable:  {trainable_params:>15,}")
    print(f"\nInference Time (ms):")
    print(f"  Mean:       {stats['mean']:>10.2f} ± {stats['std']:.2f}")
    print(f"  Median:     {stats['median']:>10.2f}")
    print(f"  Min:        {stats['min']:>10.2f}")
    print(f"  Max:        {stats['max']:>10.2f}")
    print(f"  95th %ile:  {stats['p95']:>10.2f}")
    print(f"  99th %ile:  {stats['p99']:>10.2f}")
    print(f"\nThroughput:")
    print(f"  FPS:        {1000/stats['mean']:>10.2f}")
    print(f"{'='*60}")


def main():
    # Configuration
    INPUT_HEIGHT = 288
    INPUT_WIDTH = 512
    BATCH_SIZE = 1
    NUM_WARMUP = 20
    NUM_ITERATIONS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    input_shape = (BATCH_SIZE, 9, INPUT_HEIGHT, INPUT_WIDTH)
    print(f"Input shape: {input_shape}")
    print(f"Warmup iterations: {NUM_WARMUP}")
    print(f"Benchmark iterations: {NUM_ITERATIONS}")
    
    results = {}
    
    # Benchmark Original VGG Model
    if HAS_ORIGINAL:
        print("\n" + "="*60)
        print("BENCHMARKING: TrackNetV4 (VGG Backbone)")
        print("="*60)
        try:
            model_vgg = TrackNetV4(INPUT_HEIGHT, INPUT_WIDTH, fusion_layer_type="TypeA")
            params_vgg = count_parameters(model_vgg)
            stats_vgg = benchmark_model(
                model_vgg, input_shape, NUM_WARMUP, NUM_ITERATIONS, DEVICE
            )
            results['VGG'] = (stats_vgg, params_vgg)
            print_benchmark_results("TrackNetV4 (VGG Backbone)", stats_vgg, params_vgg)
        except Exception as e:
            print(f"Error benchmarking VGG model: {e}")
    
    # Benchmark CSPNeXt Models with different configurations
    configs = [
        ("CSPNeXt (nano)", 0.33, 0.25),
        ("CSPNeXt (tiny)", 0.33, 0.375),
        ("CSPNeXt (small)", 0.33, 0.5),
        ("CSPNeXt (medium)", 0.67, 0.75),
        ("CSPNeXt (large)", 1.0, 1.0),
    ]
    
    for name, deepen, widen in configs:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {name} (deepen={deepen}, widen={widen})")
        print(f"{'='*60}")
        try:
            model = TrackNetV4_CSPNeXt_BallTracking(
                INPUT_HEIGHT, INPUT_WIDTH,
                fusion_layer_type="TypeA",
                deepen_factor=deepen,
                widen_factor=widen
            )
            params = count_parameters(model)
            stats = benchmark_model(
                model, input_shape, NUM_WARMUP, NUM_ITERATIONS, DEVICE
            )
            results[name] = (stats, params)
            print_benchmark_results(name, stats, params)
            
            # Clean up
            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Benchmark EfficientUNet Models with different configurations
    eunet_configs = [
        ("EfficientUNet (Lite)", 0.75, 0.75),
        ("EfficientUNet (B0)", 1.0, 1.0),
        ("EfficientUNet (B1)", 1.0, 1.1),
    ]
    
    for name, width_mult, depth_mult in eunet_configs:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {name} (width={width_mult}, depth={depth_mult})")
        print(f"{'='*60}")
        try:
            model = TrackNetV4_EfficientUNet(
                INPUT_HEIGHT, INPUT_WIDTH,
                fusion_layer_type="TypeA",
                width_mult=width_mult,
                depth_mult=depth_mult
            )
            params = count_parameters(model)
            stats = benchmark_model(
                model, input_shape, NUM_WARMUP, NUM_ITERATIONS, DEVICE
            )
            results[name] = (stats, params)
            print_benchmark_results(name, stats, params)
            
            # Clean up
            del model
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison Summary
    if len(results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'Params (M)':<15} {'Mean (ms)':<15} {'FPS':<10} {'Speedup'}")
        print("-"*80)
        
        baseline_time = None
        if 'VGG' in results:
            baseline_time = results['VGG'][0]['mean']
        elif results:
            baseline_time = list(results.values())[0][0]['mean']
        
        for name, (stats, params) in results.items():
            params_m = params[0] / 1e6
            fps = 1000 / stats['mean']
            speedup = baseline_time / stats['mean'] if baseline_time else 1.0
            print(f"{name:<25} {params_m:<15.2f} {stats['mean']:<15.2f} {fps:<10.2f} {speedup:.2f}x")
        
        print("="*80)
    
    # Memory usage (if CUDA)
    if DEVICE == 'cuda':
        print("\n" + "="*60)
        print("GPU MEMORY USAGE")
        print("="*60)
        for name, (stats, params) in results.items():
            print(f"\n{name}:")
            try:
                # Recreate model to measure memory
                if name == 'VGG' and HAS_ORIGINAL:
                    model = TrackNetV4(INPUT_HEIGHT, INPUT_WIDTH, fusion_layer_type="TypeA")
                elif name.startswith('CSPNeXt'):
                    # Extract config from name
                    config_map = {
                        "CSPNeXt (nano)": (0.33, 0.25),
                        "CSPNeXt (tiny)": (0.33, 0.375),
                        "CSPNeXt (small)": (0.33, 0.5),
                        "CSPNeXt (medium)": (0.67, 0.75),
                        "CSPNeXt (large)": (1.0, 1.0),
                    }
                    deepen, widen = config_map[name]
                    model = TrackNetV4_CSPNeXt_BallTracking(
                        INPUT_HEIGHT, INPUT_WIDTH,
                        fusion_layer_type="TypeA",
                        deepen_factor=deepen,
                        widen_factor=widen
                    )
                elif name.startswith('EfficientUNet'):
                    # Extract config from name
                    eunet_config_map = {
                        "EfficientUNet (Lite)": (0.75, 0.75),
                        "EfficientUNet (B0)": (1.0, 1.0),
                        "EfficientUNet (B1)": (1.0, 1.1),
                    }
                    width_mult, depth_mult = eunet_config_map[name]
                    model = TrackNetV4_EfficientUNet(
                        INPUT_HEIGHT, INPUT_WIDTH,
                        fusion_layer_type="TypeA",
                        width_mult=width_mult,
                        depth_mult=depth_mult
                    )
                
                model = model.to(DEVICE)
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    dummy_input = torch.randn(input_shape).to(DEVICE)
                    _ = model(dummy_input)
                    torch.cuda.synchronize()
                
                allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
                
                print(f"  Allocated: {allocated:.2f} MB")
                print(f"  Reserved:  {reserved:.2f} MB")
                
                del model, dummy_input
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error measuring memory: {e}")
        
        print("="*60)


if __name__ == "__main__":
    main()