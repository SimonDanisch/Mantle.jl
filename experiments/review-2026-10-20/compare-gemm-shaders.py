"""Compare disassemblies from gemm-generator-probe.jl; does not rewrite runtime shaders."""
from collections import Counter
from pathlib import Path
import re
import sys


def instructions(path):
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith(";")]
    uses = Counter(token for line in lines
                   for token in re.findall(r"%\w+", line.split("=", 1)[-1]))
    kept = []
    for line in lines:
        result = re.match(r"(%\w+) = OpAccessChain\b", line)
        if result and not uses[result[1]]:
            continue  # Unused pointer arithmetic neither reads nor writes memory.
        kept.append(line)
    first = next(i for i, line in enumerate(kept) if re.search(r"= OpFunction\b", line))
    return sorted(kept[:first]), kept[first:]


def compare(before, after):
    paths = sorted(before.glob("*.txt"))
    if not paths or {p.name for p in paths} != {p.name for p in after.glob("*.txt")}:
        raise ValueError("Expected the same nonempty set of shader disassemblies")
    different = [p.name for p in paths if instructions(p) != instructions(after / p.name)]
    print(f"Compared {len(paths)} modules; differing live instructions/declarations: {different}")
    return bool(different)


if __name__ == "__main__":
    sys.exit(compare(Path(sys.argv[1]), Path(sys.argv[2])))
