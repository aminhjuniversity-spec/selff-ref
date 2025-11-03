#!/usr/bin/env python3
"""
Smart model downloader - checks for existing local copies and HuggingFace cache.
Run this on the LOGIN NODE (which has internet access).
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("Smart Model Downloader for ExpLICD")
print("=" * 80)
print()

# Paths
project_dir = Path("/project/def-arashmoh/shahab33/Medsam/selff-ref")
local_biomedclip = project_dir / "checkpoints/BiomedCLIP"
hf_cache = Path.home() / "scratch/huggingface/hub"

print(f"Project directory: {project_dir}")
print(f"HuggingFace cache: {hf_cache}")
print()

# Check if we have local BiomedCLIP
has_local_biomedclip = (local_biomedclip / "open_clip_pytorch_model.bin").exists()

if has_local_biomedclip:
    print("✓ Found BiomedCLIP in local checkpoints/")
    print(f"  Location: {local_biomedclip}")
    print()
    
    # Offer to use it
    use_local = input("Use existing local BiomedCLIP? [Y/n]: ").strip().lower()
    
    if use_local in ['', 'y', 'yes']:
        print()
        print("Setting up HuggingFace cache to use local copy...")
        
        # Create cache structure
        cache_dir = hf_cache / "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/main"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks
        files_to_link = [
            "open_clip_config.json",
            "open_clip_pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json"
        ]
        
        for filename in files_to_link:
            src = local_biomedclip / filename
            dst = cache_dir / filename
            if src.exists():
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
                print(f"  ✓ Linked: {filename}")
        
        # Create refs
        refs_dir = cache_dir.parent.parent / "refs"
        refs_dir.mkdir(exist_ok=True)
        (refs_dir / "main").write_text("main")
        
        print("  ✓ BiomedCLIP ready in HuggingFace cache")
        print()
    else:
        print()
        print("Will download BiomedCLIP from HuggingFace...")
        has_local_biomedclip = False

print("=" * 80)
print("Downloading Required Models")
print("=" * 80)
print()

try:
    # Model 1: BiomedCLIP (only if not using local)
    if not has_local_biomedclip or use_local not in ['', 'y', 'yes']:
        print("1. Downloading BiomedCLIP...")
        print("   Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        print("   Size: ~1.5 GB")
        print("   This may take 5-10 minutes...")
        
        from open_clip import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        print("   ✓ BiomedCLIP downloaded successfully")
    else:
        print("1. BiomedCLIP")
        print("   ✓ Using local copy (skipped download)")
    print()
    
    # Model 2: BiomedBERT
    print("2. Downloading BiomedBERT config...")
    print("   Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    print("   Size: ~500 MB")
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        cache_dir=str(hf_cache)
    )
    print("   ✓ BiomedBERT config downloaded successfully")
    print()
    
    # Model 3: ViT
    print("3. Downloading Vision Transformer (ViT)...")
    print("   Model: vit_base_patch16_224.orig_in21k")
    print("   Size: ~350 MB")
    
    import timm
    vit = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)
    print("   ✓ ViT model downloaded successfully")
    print()
    
    print("=" * 80)
    print("✓ ALL MODELS READY!")
    print("=" * 80)
    print()
    print("Summary:")
    if has_local_biomedclip and use_local in ['', 'y', 'yes']:
        print("  - BiomedCLIP: Using local copy (linked to cache)")
    else:
        print("  - BiomedCLIP: Downloaded to cache")
    print("  - BiomedBERT: Downloaded to cache")
    print("  - ViT: Downloaded to cache")
    print()
    print("You can now submit your SLURM job:")
    print("  sbatch scripts/test_option1_selfrefine.sh")
    print()

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure you're on a LOGIN NODE (not compute node)")
    print("  2. Check internet connection")
    print("  3. Verify you have space in ~/scratch/")
    print(f"\nYour scratch space:")
    os.system(f"df -h {Path.home()}/scratch/")
    sys.exit(1)
