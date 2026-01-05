#!/usr/bin/env python3
"""
Migration script to organize legacy extraction results.
Moves flat JSON files to legacy folder before running new pipeline.
"""

from pathlib import Path
import shutil
import json

def migrate_legacy_files():
    """Move old flat JSON files to legacy folder."""
    
    processed_dir = Path("data/processed")
    legacy_dir = processed_dir / "legacy"
    
    if not processed_dir.exists():
        print("No processed directory found. Nothing to migrate.")
        return
    
    # Find all JSON files directly in processed/ (not in subdirectories)
    legacy_files = [f for f in processed_dir.glob("*.json") if f.is_file()]
    
    if not legacy_files:
        print("No legacy files found in data/processed/")
        return
    
    print(f"Found {len(legacy_files)} legacy files to migrate:")
    for f in legacy_files:
        print(f"  - {f.name}")
    
    # Create legacy directory
    legacy_dir.mkdir(exist_ok=True)
    
    # Move files
    moved = 0
    for file_path in legacy_files:
        dest = legacy_dir / file_path.name
        try:
            shutil.move(str(file_path), str(dest))
            print(f"✓ Moved: {file_path.name}")
            moved += 1
        except Exception as e:
            print(f"✗ Failed to move {file_path.name}: {e}")
    
    print(f"\nMigration complete: {moved}/{len(legacy_files)} files moved to {legacy_dir}")
    
    # Show summary of legacy data
    print("\n" + "="*70)
    print("LEGACY DATA SUMMARY")
    print("="*70)
    
    legacy_by_doc = {}
    for file_path in legacy_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc = data.get("document", "Unknown")
            method = data.get("method", "Unknown")
            items = data.get("item_count", 0)
            time_sec = data.get("execution_time_seconds", 0)
            
            if doc not in legacy_by_doc:
                legacy_by_doc[doc] = []
            legacy_by_doc[doc].append({
                "method": method,
                "items": items,
                "time": time_sec
            })
        except:
            pass
    
    for doc, methods in sorted(legacy_by_doc.items()):
        print(f"\n{doc}:")
        for m in sorted(methods, key=lambda x: x["method"]):
            print(f"  {m['method']:10s}: {m['items']:3d} items, {m['time']:.1f}s")
    
    print("\nLegacy files are preserved in: " + str(legacy_dir))
    print("You can now run: python run_extraction.py")

if __name__ == "__main__":
    print("="*70)
    print("LEGACY FILE MIGRATION")
    print("="*70)
    print("\nThis script will move old extraction results to data/processed/legacy/")
    print("This keeps them for reference while allowing new organized runs.\n")
    
    response = input("Proceed with migration? [y/N]: ").strip().lower()
    
    if response in ['y', 'yes']:
        migrate_legacy_files()
    else:
        print("\nMigration cancelled.")
