import os
import re

ROOT_DIR = "."  # change if needed, but project root is usually "."
TARGET_FILES = []

# Walk through repo to find .py files in agents/ and framework
for root, _, files in os.walk(ROOT_DIR):
    for f in files:
        if f.endswith(".py") and (
            "agents" in root or f == "ssff_framework.py"
        ):
            TARGET_FILES.append(os.path.join(root, f))

for file_path in TARGET_FILES:
    with open(file_path, "r") as f:
        content = f.read()

    original_content = content

    # Replace default model string
    content = re.sub(
        r'def __init__\(self, model="gpt-4o-mini"\)',
        'def __init__(self, model=DEFAULT_MODEL)',
        content,
    )

    # Add import if missing
    if "def __init__(self, model=DEFAULT_MODEL)" in content and "from config import DEFAULT_MODEL" not in content:
        content = f"from config import DEFAULT_MODEL\n\n{content}"

    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"✅ Updated {file_path}")
    else:
        print(f"⚠️ No change in {file_path}")
