import torch
import os
import argparse
import json
from datetime import datetime
from config import Config
from data_loader import load_data
import subprocess

def run_stage(stage_num, config):
    """Run a specific training stage"""
    stage_scripts = {
        1: 'train_stage1.py',
        2: 'train_stage2.py',
        3: 'train_stage3.py'
    }
    
    if stage_num not in stage_scripts:
        raise ValueError(f"Invalid stage number: {stage_num}")
    
    script = stage_scripts[stage_num]
    print(f"\n{'='*70}")
    print(f"RUNNING STAGE {stage_num}: {script}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(['python', script], check=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Dual Encoder Chatbot Training Pipeline')
    parser.add_argument('--stages', type=str, default='all', 
                       help='Stages to run: "all", "1", "2", "3", "1,2", etc.')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    
    args = parser.parse_args()
    config = Config()
    
    # Parse stages to run
    if args.stages.lower() == 'all':
        stages_to_run = [1, 2, 3]
    else:
        stages_to_run = [int(s.strip()) for s in args.stages.split(',')]
    
    print("="*70)
    print("DUAL ENCODER CHATBOT TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages to run: {stages_to_run}")
    print(f"Device: {config.device}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print("="*70)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Check for existing checkpoints
    if args.resume:
        print("\nChecking for existing checkpoints...")
        stage1_exists = os.path.exists(f"{config.checkpoint_dir}/stage1_encoders")
        stage2_exists = os.path.exists(f"{config.checkpoint_dir}/stage2_fusion")
        stage3_exists = os.path.exists(f"{config.checkpoint_dir}/stage3_decoders")
        
        if stage1_exists:
            print("✓ Found Stage 1 checkpoints")
        if stage2_exists:
            print("✓ Found Stage 2 checkpoints")
        if stage3_exists:
            print("✓ Found Stage 3 checkpoints")
    
    # Run training stages
    success = True
    for stage in stages_to_run:
        try:
            success = run_stage(stage, config)
            if not success:
                print(f"\n✗ Stage {stage} failed!")
                break
        except Exception as e:
            print(f"\n✗ Error in Stage {stage}: {str(e)}")
            success = False
            break
    
    if not success:
        print("\n" + "="*70)
        print("TRAINING FAILED")
        print("="*70)
        return
    
    # Run evaluation if requested
    if args.eval:
        print("\n" + "="*70)
        print("RUNNING EVALUATION")
        print("="*70)
        try:
            subprocess.run(['python', 'evaluate.py'], check=True)
        except Exception as e:
            print(f"✗ Evaluation failed: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCheckpoints saved in: {config.checkpoint_dir}/")
    if args.eval:
        print(f"Results saved in: {config.results_dir}/")
    
    print("\n" + "="*70)
    print("QUICK START GUIDE")
    print("="*70)
    print("To use your trained model:")
    print("1. Load encoders from: stage1_encoders/")
    print("2. Load fusion from: stage2_fusion/")
    print("3. Load decoders from: stage3_decoders/")
    print("\nSee evaluate.py for example loading code")
    print("="*70)

if __name__ == "__main__":
    main()