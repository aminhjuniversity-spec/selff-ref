#!/usr/bin/env python3
"""
ALL-IN-ONE Model Setup Script (Fixed)
Downloads all models to checkpoints/ and sets up cache.
Run this on the LOGIN NODE.
"""

import os
import sys
from pathlib import Path
import subprocess

print("=" * 80)
print("🚀 ALL-IN-ONE Model Setup for ExpLICD")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
project_dir = Path("/project/def-arashmoh/shahab33/Medsam/selff-ref")
checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

hf_cache = Path.home() / "scratch/huggingface/hub"
torch_cache = Path.home() / ".cache/torch/hub/checkpoints"

print(f"Project: {project_dir}")
print(f"Checkpoints: {checkpoint_dir}")
print()

# DON'T set HF_HOME here - let libraries use default cache first
print("=" * 80)
print("STEP 1: Downloading Models to checkpoints/")
print("=" * 80)
print()

try:
    # ========================================================================
    # MODEL 1: BiomedCLIP
    # ========================================================================
    biomedclip_dir = checkpoint_dir / "BiomedCLIP"
    
    print("[1/4] BiomedCLIP")
    if (biomedclip_dir / "open_clip_pytorch_model.bin").exists():
        print("      ✓ Already exists in checkpoints/BiomedCLIP/")
        size = (biomedclip_dir / 'open_clip_pytorch_model.bin').stat().st_size / (1024**3)
        print(f"      Size: {size:.2f} GB")
    else:
        print("      Downloading to HuggingFace cache first...")
        print("      Size: ~1.5 GB (may take 5-10 minutes)")
        
        from open_clip import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        print("      ✓ Downloaded successfully")
    print()
    
    # ========================================================================
    # MODEL 2: BiomedBERT
    # ========================================================================
    print("[2/4] BiomedBERT")
    print("      Downloading to HuggingFace cache...")
    print("      Size: ~500 MB")
    
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    
    config = AutoConfig.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    
    print("      ✓ Downloaded successfully")
    print()
    
    # ========================================================================
    # MODEL 3: ViT
    # ========================================================================
    print("[3/4] Vision Transformer (ViT)")
    print("      Downloading to torch cache...")
    print("      Size: ~343 MB")
    
    import timm
    vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)
    
    print("      ✓ Downloaded successfully")
    print()
    
    # ========================================================================
    # MODEL 4: ExpLICD checkpoint
    # ========================================================================
    explicd_checkpoint = checkpoint_dir / "explicd_best.pth"
    
    print("[4/4] ExpLICD checkpoint")
    if explicd_checkpoint.exists():
        print("      ✓ Found: checkpoints/explicd_best.pth")
        size = explicd_checkpoint.stat().st_size / (1024**3)
        print(f"      Size: {size:.2f} GB")
    else:
        print("      ✗ NOT FOUND: checkpoints/explicd_best.pth")
        print("      Please download from: https://github.com/yhygao/Explicd")
    print()
    
    # ========================================================================
    # STEP 2: Copy to checkpoints/
    # ========================================================================
    print("=" * 80)
    print("STEP 2: Organizing Models in checkpoints/")
    print("=" * 80)
    print()
    
    # Copy BiomedBERT to checkpoints
    print("[1/3] Copying BiomedBERT to checkpoints/...")
    biomedbert_dir = checkpoint_dir / "BiomedBERT"
    biomedbert_dir.mkdir(exist_ok=True)
    
    if (hf_cache / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract").exists():
        import shutil
        src = hf_cache / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"
        # Find files and copy
        for file in src.rglob("*"):
            if file.is_file() and file.suffix in ['.json', '.bin', '.txt', '.safetensors']:
                dst = biomedbert_dir / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
        print("      ✓ BiomedBERT copied to checkpoints/BiomedBERT/")
    print()
    
    # Copy ViT to checkpoints
    print("[2/3] Copying ViT to checkpoints/...")
    vit_dir = checkpoint_dir / "ViT"
    vit_dir.mkdir(exist_ok=True)
    
    torch_cache_path = Path.home() / ".cache/torch/hub/checkpoints"
    if torch_cache_path.exists():
        import shutil
        for file in torch_cache_path.glob("*vit*.safetensors"):
            dst = vit_dir / file.name
            if not dst.exists():
                shutil.copy2(file, dst)
        print("      ✓ ViT copied to checkpoints/ViT/")
    print()
    
    # BiomedCLIP already in checkpoints
    print("[3/3] BiomedCLIP...")
    if biomedclip_dir.exists():
        print("      ✓ Already in checkpoints/BiomedCLIP/")
    print()
    
    # ========================================================================
    # STEP 3: Setup Cache Links
    # ========================================================================
    print("=" * 80)
    print("STEP 3: Setting Up Cache to Use checkpoints/")
    print("=" * 80)
    print()
    
    # Setup BiomedCLIP cache
    print("[1/3] Linking BiomedCLIP cache...")
    biomedclip_cache = hf_cache / "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/main"
    biomedclip_cache.mkdir(parents=True, exist_ok=True)
    
    if biomedclip_dir.exists():
        for file in biomedclip_dir.glob("*"):
            if file.is_file():
                dst = biomedclip_cache / file.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(file)
        
        refs_dir = biomedclip_cache.parent.parent / "refs"
        refs_dir.mkdir(exist_ok=True)
        (refs_dir / "main").write_text("main")
        print("      ✓ HF cache → checkpoints/BiomedCLIP/")
    print()
    
    # Setup BiomedBERT cache
    print("[2/3] Linking BiomedBERT cache...")
    biomedbert_cache = hf_cache / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/main"
    biomedbert_cache.mkdir(parents=True, exist_ok=True)
    
    if biomedbert_dir.exists():
        for file in biomedbert_dir.glob("*"):
            if file.is_file():
                dst = biomedbert_cache / file.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(file)
        
        refs_dir = biomedbert_cache.parent.parent / "refs"
        refs_dir.mkdir(exist_ok=True)
        (refs_dir / "main").write_text("main")
        print("      ✓ HF cache → checkpoints/BiomedBERT/")
    print()
    
    # Setup ViT cache
    print("[3/3] Linking ViT cache...")
    torch_cache.mkdir(parents=True, exist_ok=True)
    
    if vit_dir.exists():
        for file in vit_dir.glob("*"):
            if file.is_file():
                dst = torch_cache / file.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(file)
        print("      ✓ Torch cache → checkpoints/ViT/")
    print()
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    print("=" * 80)
    print("✅ VERIFICATION")
    print("=" * 80)
    print()
    
    print("Checking checkpoints/ structure:")
    os.chdir(checkpoint_dir)
    
    checks = {
        'BiomedCLIP': biomedclip_dir.exists(),
        'BiomedBERT': biomedbert_dir.exists(),
        'ViT': vit_dir.exists(),
        'explicd_best.pth': explicd_checkpoint.exists()
    }
    
    for name, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
    
    print()
    
    # Show sizes
    print("Total size of checkpoints/:")
    result = subprocess.run(['du', '-sh', str(checkpoint_dir)], 
                          capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")
    print()
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("=" * 80)
    print("🎉 SUCCESS! All Models Ready!")
    print("=" * 80)
    print()
    print("Your checkpoints/ structure:")
    print("  checkpoints/")
    print("  ├── BiomedCLIP/          (Vision-language model)")
    print("  ├── BiomedBERT/          (Text encoder)")
    print("  ├── ViT/                 (Vision transformer)")
    print("  └── explicd_best.pth     (ExpLICD weights)")
    print()
    print("Cache is configured to use checkpoints/")
    print("  ~/scratch/huggingface/hub/ → checkpoints/ (symlinks)")
    print("  ~/.cache/torch/hub/       → checkpoints/ (symlinks)")
    print()
    print("=" * 80)
    print("🚀 Next Step: Run Your Test!")
    print("=" * 80)
    print()
    print("  sbatch scripts/test_option1_selfrefine.sh")
    print()
    
    if not explicd_checkpoint.exists():
        print("⚠️  WARNING: explicd_best.pth not found!")
        print("   Download it from: https://github.com/yhygao/Explicd")
        print("   Put it in: checkpoints/explicd_best.pth")
        print()

except Exception as e:
    print()
    print("=" * 80)
    print("❌ ERROR")
    print("=" * 80)
    print(f"\n{e}\n")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  1. Make sure you're on a LOGIN NODE")
    print("  2. Check internet connection")
    print("  3. Check available space")
    print()
    sys.exit(1)
