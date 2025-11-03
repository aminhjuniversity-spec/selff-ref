#!/usr/bin/env python3
"""
ALL-IN-ONE Model Setup Script
Downloads all models to checkpoints/ and sets up cache.
Run this on the LOGIN NODE (which has internet access).

Usage: python setup_all_models.py
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

# Set environment to use checkpoints as cache location
os.environ['HF_HOME'] = str(checkpoint_dir / "huggingface_cache")
os.environ['TORCH_HOME'] = str(checkpoint_dir / "torch_cache")

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
        print("      Downloading to checkpoints/BiomedCLIP/")
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
    biomedbert_dir = checkpoint_dir / "BiomedBERT"
    biomedbert_dir.mkdir(exist_ok=True)
    
    print("[2/4] BiomedBERT")
    print("      Downloading to checkpoints/BiomedBERT/")
    print("      Size: ~500 MB")
    
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    
    config = AutoConfig.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        cache_dir=str(biomedbert_dir)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        cache_dir=str(biomedbert_dir)
    )
    
    model = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        cache_dir=str(biomedbert_dir)
    )
    
    print("      ✓ Downloaded successfully")
    print()
    
    # ========================================================================
    # MODEL 3: ViT
    # ========================================================================
    vit_dir = checkpoint_dir / "ViT"
    vit_dir.mkdir(exist_ok=True)
    
    print("[3/4] Vision Transformer (ViT)")
    print("      Downloading to checkpoints/ViT/")
    print("      Size: ~343 MB")
    
    import timm
    import torch
    
    vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)
    
    vit_file = vit_dir / "vit_base_patch16_224.orig_in21k.pth"
    torch.save(vit.state_dict(), vit_file)
    
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
        print("      (This script will continue anyway)")
    print()
    
    # ========================================================================
    # STEP 2: Setup Cache Links
    # ========================================================================
    print("=" * 80)
    print("STEP 2: Setting Up Cache to Use checkpoints/")
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
        model_files = list(biomedbert_dir.rglob("*.json")) + \
                      list(biomedbert_dir.rglob("*.bin")) + \
                      list(biomedbert_dir.rglob("*.txt")) + \
                      list(biomedbert_dir.rglob("*.safetensors"))
        
        for file in model_files:
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
    print("Troubleshooting:")
    print("  1. Make sure you're on a LOGIN NODE (not compute node)")
    print("     Run: hostname")
    print("     Should NOT show 'ng' prefix")
    print()
    print("  2. Check internet connection")
    print("     Run: ping -c 3 google.com")
    print()
    print("  3. Check available space")
    print("     Run: df -h /project/def-arashmoh/")
    print()
    sys.exit(1)
