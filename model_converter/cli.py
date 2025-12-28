"""
Command-line interface for Omni Model Converter.
"""

import argparse
import sys
from pathlib import Path

from .converter import OmniModelConverter, Track


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract components (VE, AE, LLM) from unified Omni models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all components from Track B model
  python convert_model.py \\
      --input /path/to/omni/model \\
      --output /path/to/output \\
      --track b

  # Extract only LLM from Track A model
  python convert_model.py \\
      --input /path/to/ve_llm/model \\
      --output /path/to/output \\
      --track a \\
      --components llm

  # Extract VE and LLM with verification
  python convert_model.py \\
      --input /path/to/model \\
      --output /path/to/output \\
      --track b \\
      --components ve llm \\
      --verify
"""
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input unified model directory"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory"
    )

    parser.add_argument(
        "--track", "-t",
        type=str,
        choices=["a", "b"],
        default="b",
        help="Model track: 'a' (VE+LLM) or 'b' (VE+AE+LLM). Default: b"
    )

    parser.add_argument(
        "--components", "-c",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "ve", "ae", "llm", "vd", "ad"],
        help="Components to extract. Default: all"
    )

    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Maximum shard size for LLM (e.g., '5GB', '2GB'). Default: 5GB"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify extracted components after conversion"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading weights. Default: cpu"
    )

    return parser.parse_args()


def parse_size(size_str: str) -> int:
    """Parse size string like '5GB' to bytes."""
    size_str = size_str.strip().upper()

    # Check longer units first to avoid "B" matching before "GB"
    units = [
        ("TB", 1024 * 1024 * 1024 * 1024),
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
        ("B", 1),
    ]

    for unit, multiplier in units:
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)])
            return int(number * multiplier)

    # Assume bytes if no unit
    return int(size_str)


def main():
    args = parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    if not (input_path / "config.json").exists():
        print(f"Error: config.json not found in {args.input}")
        sys.exit(1)

    if not (input_path / "model.safetensors.index.json").exists():
        print(f"Error: model.safetensors.index.json not found in {args.input}")
        sys.exit(1)

    # Parse track
    track = Track.A if args.track.lower() == "a" else Track.B

    # Validate components for track
    if "ae" in args.components and track == Track.A:
        print("Warning: Audio encoder (ae) is not available for Track A")
        args.components = [c for c in args.components if c != "ae"]

    if "vd" in args.components and track == Track.A:
        print("Warning: Vision decoder (vd) is not available for Track A")
        args.components = [c for c in args.components if c != "vd"]

    if "ad" in args.components and track == Track.A:
        print("Warning: Audio decoder (ad) is not available for Track A")
        args.components = [c for c in args.components if c != "ad"]

    # Parse shard size
    max_shard_size = parse_size(args.max_shard_size)

    # Create converter and run
    print("=" * 60)
    print("Omni Model Converter")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Track:  {track.value.upper()}")
    print(f"Components: {args.components}")
    print("=" * 60)

    try:
        converter = OmniModelConverter(args.input, track)
        results = converter.convert(
            args.output,
            components=args.components,
            max_shard_size=max_shard_size
        )

        if args.verify:
            success = converter.verify(args.output)
            if not success:
                print("\nVerification failed!")
                sys.exit(1)
            print("\nVerification passed!")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nConversion completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
