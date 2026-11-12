"""
Test GPU availability and CUDA functionality for the project.
"""

import torch
import sys

print("="*70)
print("GPU AVAILABILITY TEST")
print("="*70)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU IS AVAILABLE!")
    print(f"\nCUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
    
    # Test tensor operations on GPU
    print("\n" + "="*70)
    print("TESTING GPU OPERATIONS")
    print("="*70)
    
    try:
        # Create tensors
        print("\n1. Creating tensors on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        print("   ✓ Tensors created successfully")
        
        # Matrix multiplication
        print("\n2. Testing matrix multiplication...")
        z = torch.mm(x, y)
        print(f"   ✓ Result shape: {z.shape}")
        print(f"   ✓ Result device: {z.device}")
        
        # Test model on GPU
        print("\n3. Testing neural network on GPU...")
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).cuda()
        
        input_tensor = torch.randn(32, 100).cuda()
        output = model(input_tensor)
        print(f"   ✓ Model output shape: {output.shape}")
        print(f"   ✓ Model device: {next(model.parameters()).device}")
        
        # Memory info
        print("\n4. GPU Memory Status:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
        
        print("\n" + "="*70)
        print("✅ ALL GPU TESTS PASSED!")
        print("="*70)
        print("\nYour system is ready to train on GPU! 🚀")
        
    except Exception as e:
        print(f"\n❌ ERROR during GPU testing: {e}")
        sys.exit(1)
        
else:
    print("\n❌ NO GPU AVAILABLE")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA drivers not installed")
    print("  3. PyTorch CPU-only version installed")
    print("\nTo install PyTorch with CUDA support:")
    print("  Visit: https://pytorch.org/get-started/locally/")
    print("\n  For CUDA 11.8:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  For CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device will be: cuda")
else:
    print(f"Device will be: cpu")
print("="*70)

