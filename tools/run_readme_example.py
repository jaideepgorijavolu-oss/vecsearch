"""Extract the first ```python block from README.md and run it (used in CI)."""

import pathlib
import re

readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
code = re.findall(r"```python\n(.*?)```", readme, re.S)[0]
print(f"running README example ({len(code.splitlines())} lines)")
exec(compile(code, "README.md", "exec"), {})
