#!/usr/bin/env python3
"""
Quick test script to validate Portuguese dataset loading.
Tests the first few sources to ensure they are accessible.
"""

import sys
from datasets import load_dataset

# Test sources (from preprocess_datasets.py)
TEST_SOURCES = [
    {
        "name": "FineWeb2 Portuguese",
        "dataset_id": "HuggingFaceFW/fineweb-2",
        "config": "pt",
        "text_field": "text",
    },
    {
        "name": "Portuguese-PD (Public Domain)",
        "dataset_id": "PleIAs/Portuguese-PD",
        "config": None,
        "text_field": "text",
    },
    {
        "name": "BrWaC (Brazilian Web as Corpus)",
        "dataset_id": "UFRGS/brwac",
        "config": None,
        "text_field": "text",
    },
    {
        "name": "Quati (Unicamp)",
        "dataset_id": "unicamp-dl/quati",
        "config": None,
        "text_field": "text",
    },
]


def test_dataset(source):
    """Test if a dataset can be loaded and accessed"""
    print(f"\n{'='*70}")
    print(f"Testing: {source['name']}")
    print(f"Dataset ID: {source['dataset_id']}")
    print(f"{'='*70}")

    try:
        # Build load arguments
        load_args = {
            "path": source["dataset_id"],
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }

        if source["config"]:
            load_args["name"] = source["config"]

        # Load dataset
        print(f"Loading dataset...")
        dataset = load_dataset(**load_args)

        # Try to get first sample
        print(f"Fetching first sample...")
        test_iter = iter(dataset)
        sample = next(test_iter)

        # Check text field
        text_field = source["text_field"]
        available_fields = list(sample.keys())
        print(f"Available fields: {available_fields}")

        # Try to find text field
        if text_field in sample:
            text = sample[text_field]
            print(f"✓ Text field '{text_field}' found!")
            print(f"✓ Sample length: {len(str(text))} characters")
            print(f"✓ Preview: {str(text)[:200]}...")
            return True
        else:
            # Try alternative fields
            for alt_field in ['text', 'content', 'body', 'sentence', 'document']:
                if alt_field in sample:
                    text = sample[alt_field]
                    print(f"✓ Alternative field '{alt_field}' found!")
                    print(f"✓ Sample length: {len(str(text))} characters")
                    print(f"✓ Preview: {str(text)[:200]}...")
                    return True

            print(f"✗ No suitable text field found")
            return False

    except Exception as e:
        print(f"✗ Error: {str(e)[:200]}")
        return False


def main():
    print("\n" + "="*70)
    print("Testing Portuguese Dataset Sources")
    print("="*70)

    results = {}

    for source in TEST_SOURCES:
        success = test_dataset(source)
        results[source['name']] = success

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\n✓ Successful ({len(successful)}/{len(results)}):")
    for name in successful:
        print(f"  - {name}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}/{len(results)}):")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "="*70)

    if successful:
        print("✓ At least one dataset is accessible!")
        print("The preprocessing script will automatically fallback to working sources.")
        return 0
    else:
        print("✗ All datasets failed to load.")
        print("Please check your internet connection and HuggingFace access.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
