# inspect_csv.py
import pandas as pd
import csv

file_path = "/Users/mahriovezmyradova/MedicalASR-Summarization/data/outputs/full_dataset_analysis/all_transcriptions_copy.csv"

print("="*60)
print("CSV INSPECTION")
print("="*60)

# Method 1: Try reading with different parameters
print("\n1. Trying different pandas read_csv parameters:")
try:
    # Try with python engine and error handling
    df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    print(f"✅ Success with python engine! Loaded {len(df)} rows")
    print(f"Columns found: {list(df.columns)}")
except Exception as e:
    print(f"❌ Python engine failed: {e}")

try:
    # Try with no header and inspect
    df_no_header = pd.read_csv(file_path, header=None, engine='python', nrows=10)
    print(f"\n✅ First 10 rows with no header assumption:")
    print(df_no_header.head(10))
except Exception as e:
    print(f"❌ Failed: {e}")

# Method 2: Read raw lines to see the issue
print("\n2. Reading raw first 20 lines:")
try:
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 20:  # Show first 20 lines
                print(f"Line {i+1}: {line.rstrip()}")
                # Count commas to see inconsistency
                comma_count = line.count(',')
                print(f"   Commas: {comma_count}")
            else:
                break
except Exception as e:
    print(f"❌ Failed to read file: {e}")

# Method 3: Try using csv module to parse
print("\n3. Parsing with csv module:")
try:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 10:
                print(f"Row {i+1}: {len(row)} fields - {row[:3]}...")
            else:
                break
except Exception as e:
    print(f"❌ CSV module failed: {e}")