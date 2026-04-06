import sys
from pathlib import Path

# Add project root to path so `from koi import ...` works without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
