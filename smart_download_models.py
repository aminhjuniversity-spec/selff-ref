#!/usr/bin/env python3
"""
Smart Model Downloader - Downloads ALL models to checkpoints/
Each model gets its own folder in checkpoints/
Run this on the LOGIN NODE (which has internet access).
"""

import os
import sys
from pathlib import Path
import shutil

print("=" * 80)
print("Smart Model Downloader for ExpLICD")
print("=" * 80)
print()

# Paths
project_dir = Path("/project/def-arashmoh/shahab33/Medsam/selff-ref")
checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

hf_cache = Path.home() / "scratch/huggingface/hub"
torch_cache = Path.home() / ".cache/torch/hub/checkpoints"

print(f"Project directory: {project_dir}")
print(f"Checkpoints directory: {checkpoint_dir}")
print(f"HuggingFace cache: {hf_cache}")
print()

print("=" * 80)
print("Downloading Models to checkpoints/")
print("=" * 80)
print()

try:
    # ========================================================================
    # MODEL 1: BiomedCLIP
    # ========================================================================
    local_biomedclip = checkpoint_dir / "BiomedCLIP"
    
    print("[1/4] BiomedCLIP")
    if (local_biomedclip / "open_clip_pytorch_model.bin").exists():
        print("      ✓ Already exists in checkpoints/BiomedCLIP/")
        size = (local_biomedclip / 'open_clip_pytorch_model.bin').stat().st_size / (1024**3)
        print(f"      Size: {size:.2f} GB")
    else:
        print("      Downloading...")
        print("      Size: ~1.5 GB (may take 5-10 minutes)")
        
        from open_clip import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        print("      ✓ Downloaded to HuggingFace cache")
        
        # Now copy to checkpoints
        print("      Copying to checkpoints/BiomedCLIP/...")
        local_biomedclip.mkdir(exist_ok=True)
        biomedclip_cache = hf_cache / "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        for file in biomedclip_cache.rglob("*"):
            if file.is_file():
                dst = local_biomedclip / file.name
                if not dst.exists():
                    shutil.copy2(file, dst)
        print("      ✓ Copied to checkpoints/BiomedCLIP/")
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
    
    # Download directly to checkpoints/BiomedBERT/
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
    
    print("      ✓ Downloaded to checkpoints/BiomedBERT/")
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
    
    # Download ViT
    vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)
    
    # Save to checkpoints/ViT/
    vit_file = vit_dir / "vit_base_patch16_224.orig_in21k.pth"
    torch.save(vit.state_dict(), vit_file)
    print("      ✓ Saved to checkpoints/ViT/")
    
    # Also copy the original safetensors file if it exists
    default_torch_cache = Path.home() / ".cache/torch/hub/checkpoints"
    if default_torch_cache.exists():
        for file in default_torch_cache.glob("*vit*.safetensors"):
            dst = vit_dir / file.name
            if not dst.exists():
                shutil.copy2(file, dst)
                print(f"      ✓ Also copied: {file.name}")
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
        print("      Download from: https://github.com/yhygao/Explicd")
    print()
    
    # ========================================================================
    # STEP 2: Setup Cache to Use checkpoints/
    # ========================================================================
    print("=" * 80)
    print("Setting Up Cache to Use checkpoints/")
    print("=" * 80)
    print()
    
    # Setup BiomedCLIP cache
    print("[1/3] Linking BiomedCLIP cache → checkpoints/BiomedCLIP/")
    biomedclip_cache = hf_cache / "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/main"
    biomedclip_cache.mkdir(parents=True, exist_ok=True)
    
    for file in local_biomedclip.glob("*"):
        if file.is_file():
            dst = biomedclip_cache / file.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(file)
    
    refs_dir = biomedclip_cache.parent.parent / "refs"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "main").write_text("main")
    print("      ✓ Cache linked")
    print()
    
    # Setup BiomedBERT cache
    print("[2/3] Linking BiomedBERT cache → checkpoints/BiomedBERT/")
    biomedbert_cache = hf_cache / "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/main"
    biomedbert_cache.mkdir(parents=True, exist_ok=True)
    
    # Find all model files in BiomedBERT directory
    for file in biomedbert_dir.rglob("*"):
        if file.is_file():
            dst = biomedbert_cache / file.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(file)
    
    refs_dir = biomedbert_cache.parent.parent / "refs"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "main").write_text("main")
    print("      ✓ Cache linked")
    print()
    
    # Setup ViT cache
    print("[3/3] Linking ViT cache → checkpoints/ViT/")
    torch_cache.mkdir(parents=True, exist_ok=True)
    
    for file in vit_dir.glob("*"):
        if file.is_file():
            dst = torch_cache / file.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(file)
    print("      ✓ Cache linked")
    print()
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    print("=" * 80)
    print("✅ VERIFICATION")
    print("=" * 80)
    print()
    
    print("Checking checkpoints/ structure:")
    checks = {
        'BiomedCLIP': local_biomedclip.exists(),
        'BiomedBERT': biomedbert_dir.exists(),
        'ViT': vit_dir.exists(),
        'explicd_best.pth': explicd_checkpoint.exists()
    }
    
    for name, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
    
    print()
    print("Folder contents:")
    for folder in ['BiomedCLIP', 'BiomedBERT', 'ViT']:
        folder_path = checkpoint_dir / folder
        if folder_path.exists():
            file_count = len(list(folder_path.rglob("*")))
            print(f"  checkpoints/{folder}/ → {file_count} files")
    
    print()
    
    # ========================================================================
    # SUCCESS
    # ========================================================================
    print("=" * 80)
    print("🎉 SUCCESS! All Models in checkpoints/")
    print("=" * 80)
    print()
    print("Your checkpoints/ structure:")
    print("  checkpoints/")
    print("  ├── BiomedCLIP/          (Vision-language model)")
    print("  ├── BiomedBERT/          (Text encoder)")
    print("  ├── ViT/                 (Vision transformer)")
    print("  └── explicd_best.pth     (ExpLICD weights)")
    print()
    print("Cache is configured (symlinks → checkpoints/):")
    print("  ~/scratch/huggingface/hub/ → checkpoints/")
    print("  ~/.cache/torch/hub/       → checkpoints/")
    print()
    print("=" * 80)
    print("🚀 Ready to Run!")
    print("=" * 80)
    print()
    print("  sbatch scripts/test_option1_selfrefine.sh")
    print()
    
    if not explicd_checkpoint.exists():
        print("⚠️  Note: explicd_best.pth not found")
        print("   Download from: https://github.com/yhygao/Explicd")
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
