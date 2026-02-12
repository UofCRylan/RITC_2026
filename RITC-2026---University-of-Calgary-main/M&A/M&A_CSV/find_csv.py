from pathlib import Path
import os

print("="*80)
print("CSV FILE DIAGNOSTIC")
print("="*80)

# Check what the script thinks the path is
IMPACT_CSV = Path(r"C:\Users\rylan\Desktop\M&A\ticker_by_ticker_impacts.csv")
print(f"\n1. Looking for CSV at:")
print(f"   {IMPACT_CSV}")
print(f"   Exists? {IMPACT_CSV.exists()}")

# Check the parent directory
parent_dir = Path(r"C:\Users\rylan\Desktop\M&A")
print(f"\n2. Checking parent directory:")
print(f"   {parent_dir}")
print(f"   Exists? {parent_dir.exists()}")

# List all CSV files in that directory
if parent_dir.exists():
    print(f"\n3. CSV files in {parent_dir}:")
    csv_files = list(parent_dir.glob("*.csv"))
    if csv_files:
        for f in csv_files:
            print(f"   ✓ {f.name}")
    else:
        print("   ⚠ No CSV files found!")
    
    print(f"\n4. ALL files in {parent_dir}:")
    all_files = list(parent_dir.glob("*"))
    for f in all_files[:20]:  # Show first 20
        print(f"   - {f.name}")
else:
    print("   ✗ Directory does not exist!")

# Check current working directory
print(f"\n5. Current working directory:")
print(f"   {Path.cwd()}")

# Check if there are CSV files in current directory
cwd_csvs = list(Path.cwd().glob("*.csv"))
if cwd_csvs:
    print(f"\n6. CSV files in current directory:")
    for f in cwd_csvs:
        print(f"   ✓ {f.name}")

print("\n" + "="*80)
