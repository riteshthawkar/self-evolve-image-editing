#!/usr/bin/env python3
"""Print family counts for self-evolution manifest files."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


for arg in sys.argv[1:]:
    path = Path(arg)
    rows = 0
    primary = Counter()
    families = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            primary_value = record.get("primary_family") or metadata.get("primary_family") or "unknown"
            primary.update([primary_value])
            fams = record.get("edit_families") or metadata.get("edit_families") or []
            families.update(fams or [primary_value])
    print(path)
    print("rows", rows)
    print("primary", dict(primary.most_common(12)))
    print("families", dict(families.most_common(12)))
