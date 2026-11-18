"""测试包：把 src 加入 sys.path，使 ``python -m unittest`` 与 pytest 行为一致。"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
