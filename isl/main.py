"""
Main Entry Point for Sign Language Interpreter System
Run this script to start the interpreter in various modes.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print system banner"""
    banner = """
    Sign Language Interpreter!
    """
    print(banner)


def mode_extract(args):
    """Extract landmarks from video dataset"""
    from src.ml.extract_landmarks import extract_dataset
    
    print("\n  LANDMARK EXTRACTION MODE")
    print("  " + "=" * 50)
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  FPS:      {args.fps}")
    print("  " + "=" * 50 + "\n")
    
    extract_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_fps=args.fps,
    )


def mode_train(args):
    """Train the ML recognition model"""
    from src.ml.train import train_model
    
    print("\n  MODEL TRAINING MODE")
    print("  " + "=" * 50)
    print(f"  Data:     {args.data_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Model:    {args.model_type}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.lr}")
    print("  " + "=" * 50 + "\n")
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        seq_length=args.seq_length,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        augment_repeats=args.augment_repeats,
        mirror_augment=args.mirror_augment,
        noise_std=args.noise_std,
        frame_drop_prob=args.frame_drop_prob,
        jitter_shift=args.jitter_shift,
        time_stretch_min=args.time_stretch_min,
        time_stretch_max=args.time_stretch_max,
        temporal_crop_prob=args.temporal_crop_prob,
    )


def mode_benchmark(args):
    """Train several model families and compare validation metrics."""
    from src.ml.benchmark import benchmark

    print("\n  MODEL BENCHMARK MODE")
    print("  " + "=" * 50)
    print(f"  Data:     {args.data_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Models:   {', '.join(args.models)}")
    print(f"  Device:   {args.device}")
    print("  " + "=" * 50 + "\n")

    benchmark(args)


def mode_interpret(args):
    """Run the main real-time interpreter (default mode)"""
    from src.pipeline.fast_pipeline import FastPipeline
    # Import SignResult from whichever engine the pipeline uses
    try:
        from src.recognition.ml_engine import SignResult
    except ImportError:
        from src.recognition.sign_engine import SignResult

    print("\n  REAL-TIME INTERPRETER MODE")
    print("  " + "=" * 50)
    print(f"  Camera: {args.camera}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Pose tracking: {'off' if args.no_pose else 'on'}")
    print("  " + "=" * 50 + "\n")

    pipeline = FastPipeline(
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=30,
        use_pose=not args.no_pose,
        show_visualization=not args.no_viz,
    )

    def on_result(result: SignResult):
        if result.sign and result.is_new:
            stable = " ✓" if result.is_stable else ""
            print(f"  [{result.sign_type:>7}] {result.sign:>15} "
                  f"({result.confidence:.0%}){stable}")

    try:
        pipeline.start(on_result=on_result if args.verbose else None)
    except KeyboardInterrupt:
        print("\n  Stopped by user")
    finally:
        sentence = pipeline.engine.get_sentence()
        if sentence:
            print(f"\n  Final sentence: {sentence}")
        print(f"  Total frames: {pipeline.frame_count}")


def mode_test(args):
    """Run system tests"""
    print("\n  TEST MODE")
    print("  " + "=" * 50 + "\n")

    import pytest
    test_args = ['tests/', '-v']
    if args.coverage:
        test_args.extend(['--cov=src', '--cov-report=html'])
    pytest.main(test_args)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Sign Language Interpreter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                            # Start interpreter (default)
  python main.py interpret                  # Same as above
  python main.py extract                    # Extract landmarks from data/
  python main.py train                      # Train model on extracted landmarks
  python main.py benchmark                  # Compare hybrid/transformer/tcn/full/lite
  python main.py train --epochs 200         # Train with more epochs
  python main.py interpret --verbose        # Show detections in console
  python main.py test                       # Run tests
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Interpret mode (default)
    interp = subparsers.add_parser('interpret', help='Real-time sign language interpreter')
    interp.add_argument('--camera', type=int, default=0, help='Camera index')
    interp.add_argument('--width', type=int, default=1280, help='Camera width')
    interp.add_argument('--height', type=int, default=720, help='Camera height')
    interp.add_argument('--no-pose', action='store_true', help='Disable pose tracking')
    interp.add_argument('--no-viz', action='store_true', help='Disable visualization')
    interp.add_argument('--fast', action='store_true', help='Use lite model')
    interp.add_argument('--verbose', action='store_true', help='Print detections')

    # Extract landmarks mode
    extract = subparsers.add_parser('extract', help='Extract landmarks from video dataset')
    extract.add_argument('--data-dir', default='data', help='Path to dataset root')
    extract.add_argument('--output-dir', default='extracted_data', help='Output directory')
    extract.add_argument('--fps', type=float, default=15.0, help='Sampling FPS')

    # Train mode
    train = subparsers.add_parser('train', help='Train the recognition model')
    train.add_argument('--data-dir', default='extracted_data', help='Extracted landmarks dir')
    train.add_argument('--output-dir', default='models', help='Model output dir')
    train.add_argument('--model-type', default='hybrid', choices=['hybrid', 'transformer', 'tcn', 'full', 'lite'])
    train.add_argument('--hidden-dim', type=int, default=256)
    train.add_argument('--num-layers', type=int, default=2)
    train.add_argument('--dropout', type=float, default=0.35)
    train.add_argument('--epochs', type=int, default=150)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--lr', type=float, default=1e-3)
    train.add_argument('--min-lr', type=float, default=1e-6)
    train.add_argument('--weight-decay', type=float, default=1e-4)
    train.add_argument('--label-smoothing', type=float, default=0.05)
    train.add_argument('--no-mixup', action='store_true')
    train.add_argument('--mixup-alpha', type=float, default=0.15)
    train.add_argument('--seq-length', type=int, default=30)
    train.add_argument('--patience', type=int, default=25)
    train.add_argument('--warmup-epochs', type=int, default=10)
    train.add_argument('--val-ratio', type=float, default=0.2)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    train.add_argument('--no-amp', action='store_true')
    train.add_argument('--num-workers', type=int, default=0)
    train.add_argument('--no-augment', action='store_true')
    train.add_argument('--augment-repeats', type=int, default=2)
    train.add_argument('--mirror-augment', action='store_true')
    train.add_argument('--noise-std', type=float, default=0.012)
    train.add_argument('--frame-drop-prob', type=float, default=0.08)
    train.add_argument('--jitter-shift', type=float, default=0.025)
    train.add_argument('--time-stretch-min', type=float, default=0.85)
    train.add_argument('--time-stretch-max', type=float, default=1.18)
    train.add_argument('--temporal-crop-prob', type=float, default=0.25)

    # Benchmark mode
    bench = subparsers.add_parser('benchmark', help='Train and compare multiple model families')
    bench.add_argument('--data-dir', default='extracted_data', help='Extracted landmarks dir')
    bench.add_argument('--output-dir', default='models/benchmark', help='Benchmark output dir')
    bench.add_argument('--models', nargs='+', default=['hybrid', 'transformer', 'tcn', 'full', 'lite'],
                       choices=['hybrid', 'transformer', 'tcn', 'full', 'lite'])
    bench.add_argument('--hidden-dim', type=int, default=256)
    bench.add_argument('--num-layers', type=int, default=2)
    bench.add_argument('--dropout', type=float, default=0.35)
    bench.add_argument('--epochs', type=int, default=180)
    bench.add_argument('--batch-size', type=int, default=32)
    bench.add_argument('--lr', type=float, default=1e-3)
    bench.add_argument('--min-lr', type=float, default=1e-6)
    bench.add_argument('--weight-decay', type=float, default=1e-4)
    bench.add_argument('--label-smoothing', type=float, default=0.05)
    bench.add_argument('--no-mixup', action='store_true')
    bench.add_argument('--mixup-alpha', type=float, default=0.15)
    bench.add_argument('--seq-length', type=int, default=30)
    bench.add_argument('--patience', type=int, default=35)
    bench.add_argument('--warmup-epochs', type=int, default=10)
    bench.add_argument('--val-ratio', type=float, default=0.2)
    bench.add_argument('--seed', type=int, default=42)
    bench.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    bench.add_argument('--no-amp', action='store_true')
    bench.add_argument('--num-workers', type=int, default=0)
    bench.add_argument('--no-augment', action='store_true')
    bench.add_argument('--augment-repeats', type=int, default=3)
    bench.add_argument('--mirror-augment', action='store_true')
    bench.add_argument('--noise-std', type=float, default=0.012)
    bench.add_argument('--frame-drop-prob', type=float, default=0.08)
    bench.add_argument('--jitter-shift', type=float, default=0.025)
    bench.add_argument('--time-stretch-min', type=float, default=0.85)
    bench.add_argument('--time-stretch-max', type=float, default=1.18)
    bench.add_argument('--temporal-crop-prob', type=float, default=0.25)

    # Test mode
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage')

    args = parser.parse_args()
    print_banner()

    if args.mode == 'test':
        mode_test(args)
    elif args.mode == 'extract':
        mode_extract(args)
    elif args.mode == 'train':
        mode_train(args)
    elif args.mode == 'benchmark':
        mode_benchmark(args)
    else:
        # Default to interpret mode
        if not hasattr(args, 'camera'):
            args.camera = 0
            args.width = 1280
            args.height = 720
            args.no_pose = False
            args.no_viz = False
            args.fast = False
            args.verbose = False
        mode_interpret(args)


if __name__ == "__main__":
    main()
