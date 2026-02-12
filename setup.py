"""
Setup and Initialize Script
セットアップと初期化スクリプト

機能:
1. 必要なディレクトリの作成
2. 環境チェック
3. データファイルの存在確認
4. GPU確認
"""

import sys
from pathlib import Path
import importlib.util

print("="*60)
print("🚀 Pump Range Deviation Forecast - Setup")
print("="*60)

# 基本モジュールチェック
required_modules = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'sklearn': 'scikit-learn',
}

print("\n📦 Checking required packages...")
missing_packages = []

for module_name, package_name in required_modules.items():
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"  ❌ {package_name} not found")
        missing_packages.append(package_name)
    else:
        print(f"  ✓ {package_name}")

if missing_packages:
    print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
    print("Install them with: pip install -r requirements.txt")
    sys.exit(1)

# GPU確認
print("\n🖥️  Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ℹ️  No GPU detected. Will use CPU (slower)")
except Exception as e:
    print(f"  ⚠ Error checking GPU: {e}")

# ディレクトリ作成
print("\n📁 Creating project directories...")
try:
    from config import create_directories, SOURCE_CSV_PATH
    create_directories()
except Exception as e:
    print(f"  ⚠ Error creating directories: {e}")
    print("  Creating manually...")
    
    project_root = Path(__file__).parent
    dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "ranges",
        project_root / "models" / "granite_pump_lora",
        project_root / "results",
        project_root / "notebooks",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(project_root)}")

# ソースデータ確認
print("\n📊 Checking source data...")
try:
    from config import SOURCE_CSV_PATH
    
    if SOURCE_CSV_PATH.exists():
        size_mb = SOURCE_CSV_PATH.stat().st_size / 1e6
        print(f"  ✓ Found: {SOURCE_CSV_PATH.name} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ Not found: {SOURCE_CSV_PATH}")
        print(f"     Expected location: {SOURCE_CSV_PATH}")
        print(f"     Please ensure the CSV file is in the correct location")
except Exception as e:
    print(f"  ⚠ Error checking source data: {e}")

# 推奨次のステップ
print("\n" + "="*60)
print("✅ Setup Complete!")
print("="*60)
print("\n📌 Next Steps:")
print("  1. Run: python data_preprocessing.py")
print("  2. Run: python range_definition.py")
print("  3. Run: python train.py")
print("  4. Run: python inference.py")
print("  5. Run: python evaluate.py")
print("\nSee QUICKSTART.md for detailed instructions.")
print("="*60)
