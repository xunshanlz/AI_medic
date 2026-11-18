"""测试包。"""

import sys
from pathlib import Path

# 让测试不依赖「当前目录恰好是项目根」：把项目根加进 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
