#!/usr/bin/env python3
"""
Test script to verify Word loader optimizations for RAG indexing.
Tests text cleaning, payload size reduction, and processing efficiency.
"""
import sys
from pathlib import Path
import time
import json

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.document_loaders.word_loader import WordLoaderUniversal

def test_word_optimization():
    """Test the optimized Word loader"""

    # Test file - you'll need to provide a .docx file
    test_file = "C:/Users/JONATHAN/Downloads/API TRON - Junio 2020.docx"  # Replace with actual path

    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        print("Please update the test_file path to a valid .docx file")
        return

    print("=" * 80)
    print("🧪 TESTING WORD LOADER OPTIMIZATIONS")
    print("=" * 80)

    # Test with cleaning enabled (optimized)
    print("\n1. TESTING WITH TEXT CLEANING ENABLED (OPTIMIZED)")
    print("-" * 50)

    start_time = time.time()
    loader_clean = WordLoaderUniversal(enable_parallel=False, clean_text=True)
    result_clean = loader_clean.load(Path(test_file))
    clean_time = time.time() - start_time

    print(".2f")
    print(f"📊 Stats: {json.dumps(loader_clean.stats, indent=2)}")

    # Calculate payload size for first section
    if result_clean.sections:
        first_section = result_clean.sections[0]
        section_size = len(first_section.content.encode('utf-8'))
        metadata_size = len(json.dumps(first_section.metadata, ensure_ascii=False).encode('utf-8'))
        total_size = section_size + metadata_size

        print(f"📏 First section content size: {section_size} bytes")
        print(f"📏 First section metadata size: {metadata_size} bytes")
        print(f"📏 Total payload size: {total_size} bytes")

    # Test with cleaning disabled (original)
    print("\n2. TESTING WITH TEXT CLEANING DISABLED (ORIGINAL)")
    print("-" * 50)

    start_time = time.time()
    loader_no_clean = WordLoaderUniversal(enable_parallel=False, clean_text=False)
    result_no_clean = loader_no_clean.load(Path(test_file))
    no_clean_time = time.time() - start_time

    print(".2f")
    print(f"📊 Stats: {json.dumps(loader_no_clean.stats, indent=2)}")

    # Calculate payload size for first section
    if result_no_clean.sections:
        first_section = result_no_clean.sections[0]
        section_size = len(first_section.content.encode('utf-8'))
        metadata_size = len(json.dumps(first_section.metadata, ensure_ascii=False).encode('utf-8'))
        total_size = section_size + metadata_size

        print(f"📏 First section content size: {section_size} bytes")
        print(f"📏 First section metadata size: {metadata_size} bytes")
        print(f"📏 Total payload size: {total_size} bytes")

    # Compare results
    print("\n3. OPTIMIZATION RESULTS")
    print("-" * 50)

    time_improvement = ((no_clean_time - clean_time) / no_clean_time) * 100 if no_clean_time > 0 else 0
    print(".2f")

    sections_clean = len(result_clean.sections)
    sections_no_clean = len(result_no_clean.sections)
    print(f"📄 Sections extracted: {sections_clean} (clean) vs {sections_no_clean} (no clean)")

    # Check for base64 removal
    clean_content = result_clean.content if hasattr(result_clean, 'content') else ''
    no_clean_content = result_no_clean.content if hasattr(result_no_clean, 'content') else ''

    clean_has_base64 = 'base64' in clean_content.lower()
    no_clean_has_base64 = 'base64' in no_clean_content.lower()

    print(f"🖼️  Base64 images in content: {clean_has_base64} (clean) vs {no_clean_has_base64} (no clean)")

    # Check metadata size reduction
    if result_clean.sections and result_no_clean.sections:
        clean_metadata = json.dumps(result_clean.sections[0].metadata, ensure_ascii=False)
        no_clean_metadata = json.dumps(result_no_clean.sections[0].metadata, ensure_ascii=False)

        clean_meta_size = len(clean_metadata.encode('utf-8'))
        no_clean_meta_size = len(no_clean_metadata.encode('utf-8'))

        size_reduction = ((no_clean_meta_size - clean_meta_size) / no_clean_meta_size) * 100 if no_clean_meta_size > 0 else 0
        print(".1f")

    print("\n✅ OPTIMIZATION TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_word_optimization()