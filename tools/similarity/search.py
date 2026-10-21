"""Local semantic retrieval; scores nominate code for review, not automatic merging."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import tomllib

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "jina": ("jinaai/jina-code-embeddings-0.5b", "4db235132dafbe56a8b9c5f59b59795ecf58a4a7"),
}
PAIRS = [
    ("memory selector", "src/vulkan/runtime/memory.jl", "function find_memory_type_optional(",
     "src/vulkan/runtime/memory.jl", "function find_memory_type("),
    ("borrowed Metal array", "src/metal/memory.jl", "function deviceview(",
     "src/metal/ka.jl", "function deviceslice("),
    ("raster/mesh shared setup", "src/metal/graphics.jl", "function compile_pipeline(p::Mantle.GraphicsPipeline,",
     "src/metal/mesh.jl", "function compile_pipeline(p::Mantle.MeshPipeline,"),
    ("TLAS batch transforms", "src/metal/hwtlas.jl", "transforms::MtlVector{Mat3x4f}",
     "src/vulkan/raytracing/hwtlas.jl", "transforms::LavaArray{Mat3x4f, 1}"),
    ("TLAS single transform", "src/metal/hwtlas.jl", "function Raycore.update_transform!(",
     "src/vulkan/raytracing/hwtlas.jl",
     "Raycore.update_transform!(hwtlas::VulkanTLAS, handle::Raycore.TLASHandle, transform::Mat3x4f)"),
    ("GC pressure policy", "src/memory/pool.jl", "function autocollect!(",
     "src/vulkan/runtime/memory.jl", "function maybe_collect("),
]
QUESTIONS = [
    "Find code that chooses the first GPU memory type allowed by a bit mask and required property flags.",
    "Find code that constructs a borrowed GPU array view from an existing native Metal buffer with a byte offset and dimensions.",
    "Find implementations of garbage collection decisions based on GPU memory pressure and time spent collecting.",
    "Find code that updates transforms for a batch of ray tracing instances and marks the acceleration structure dirty.",
    "Find the Vulkan Float32 depth attachment format mapping.",
    "Find the Vulkan Float32 sampled color texture format mapping.",
]


def location(c):
    return f"{c['file']}:{c['start']}-{c['end']} ({c['kind']})"


def words(text):
    return Counter(re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\s\w]", text))


def lexical_vectors(chunks):
    counts = [words(c["code"]) for c in chunks]
    df = Counter(t for c in counts for t in c)
    vectors = []
    for c in counts:
        v = {t: (1 + np.log(n)) * np.log(1 + len(chunks) / df[t]) for t, n in c.items()}
        norm = np.sqrt(sum(x * x for x in v.values()))
        vectors.append({t: x / norm for t, x in v.items()})
    return vectors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, default="qwen")
    p.add_argument("--query")
    args = p.parse_args()
    base = Path(__file__).resolve().parent
    chunks = tomllib.loads((base / ".cache/chunks.toml").read_text())["chunks"]
    if any("declaration" not in c for c in chunks):
        raise SystemExit("Source index predates signature selectors; rerun extract.jl")
    model_id, revision = MODELS[args.model]
    identity = hashlib.sha256(json.dumps([model_id, revision, chunks], sort_keys=True).encode()).hexdigest()
    cache = base / f".cache/{args.model}-{identity[:16]}.npz"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    model = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                     attn_implementation="sdpa").to("cuda").eval()
    torch.set_num_threads(4)

    def embed(texts):
        # Sort by length to reduce padding. Batch token budget also accommodates
        # long complete methods without silently dropping their latter half.
        tokens = tokenizer(texts, truncation=False)["input_ids"]
        if max(map(len, tokens)) > 32768:
            raise ValueError("Chunk exceeds model context; split it explicitly before indexing")
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        result = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        start = time.monotonic()
        done = 0
        while done < len(order):
            ids = [order[done]]
            while done + len(ids) < len(order) and len(ids) < 16:
                nxt = order[done + len(ids)]
                if len(tokens[nxt]) * (len(ids) + 1) > 4096:
                    break
                ids.append(nxt)
            batch = tokenizer.pad({"input_ids": [tokens[i] for i in ids]}, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**batch).last_hidden_state[:, -1]
                result[ids] = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
            done += len(ids)
            if done % 160 < len(ids) or done == len(order):
                print(f"{args.model}: embedded {done}/{len(order)} in {time.monotonic()-start:.1f}s", flush=True)
        return result

    def query_prefix(text, code=False):
        if args.model == "jina":
            instruction = ("Find an equivalent code snippet given the following code snippet:\n" if code
                           else "Find the most relevant code snippet given the following query:\n")
            return instruction + text
        task = ("Given Julia code, retrieve implementations with equivalent behavior or shared algorithmic responsibilities."
                if code else "Given a description, retrieve Julia code that implements the described behavior.")
        return f"Instruct: {task}\nQuery:{text}"

    if cache.exists():
        vectors = np.load(cache)["vectors"]
        print(f"Loaded {len(vectors)} cached vectors", flush=True)
    else:
        prefix = "Candidate code snippet:\n" if args.model == "jina" else ""
        vectors = embed([prefix + c["code"] for c in chunks])
        np.savez_compressed(cache, vectors=vectors)

    def top(scores, n=10, exclude=()):
        return [int(i) for i in np.argsort(-scores) if i not in exclude][:n]

    if args.query:
        scores = embed([query_prefix(args.query)])[0] @ vectors.T
        for i in top(scores):
            print(f"{scores[i]:.4f} {location(chunks[i])}\n  {chunks[i]['signature']}")
        return

    def index(path, signature):
        matches = [i for i, c in enumerate(chunks) if c["file"] == path
                   and c["kind"] == "method"
                   and signature.removeprefix("function ") in c["declaration"]]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous calibration selector: {path}: {signature}")
        return matches[0] if matches else None

    selected = []
    for name, a, signature_a, b, signature_b in PAIRS:
        source, target = index(a, signature_a), index(b, signature_b)
        if source is None or target is None:
            print(f"Skipping removed calibration method: {name}", flush=True)
        else:
            selected.append((name, source, target))
    queries = [query_prefix(chunks[a]["code"], code=True) for _, a, _ in selected]
    qvec = embed(queries + [query_prefix(q) for q in QUESTIONS])
    lex = lexical_vectors(chunks)
    rows = []
    for k, (name, a, b) in enumerate(selected):
        scores = qvec[k] @ vectors.T
        lscores = np.array([sum(v * c.get(t, 0) for t, v in lex[a].items()) for c in lex])
        # Regions from the query method count as self-matches too.
        exclude = [i for i, c in enumerate(chunks) if c["file"] == chunks[a]["file"]
                   and c["start"] >= chunks[a]["start"] and c["end"] <= chunks[a]["end"]]
        rank = 1 + sum(scores[i] > scores[b] for i in range(len(chunks)) if i not in exclude)
        lrank = 1 + sum(lscores[i] > lscores[b] for i in range(len(chunks)) if i not in exclude)
        rows.append({"name": name, "semantic_rank": int(rank), "lexical_rank": int(lrank),
                     "score": float(scores[b]), "source": location(chunks[a]), "target": location(chunks[b]),
                     "top": [{"location": location(chunks[i]), "score": float(scores[i])}
                             for i in top(scores, exclude=exclude)]})
    query_results = []
    for k, query in enumerate(QUESTIONS):
        scores = qvec[len(selected) + k] @ vectors.T
        query_results.append({"query": query, "top": [{"location": location(chunks[i]),
                              "score": float(scores[i])} for i in top(scores)]})
    # Symmetric document similarity supplies a broad all-pairs candidate list.
    # Query-prefixed retrieval above is evaluated separately and used for search.
    similarities = vectors @ vectors.T
    candidates = []
    for a in range(len(chunks)):
        for b in top(similarities[a], 12, exclude=(a,)):
            if a < b and chunks[a]["file"] != chunks[b]["file"]:
                candidates.append((float(similarities[a, b]), a, b))
    candidates.sort(reverse=True)
    report = {"model": model_id, "revision": revision, "index_sha256": identity,
              "chunks": len(chunks), "calibration": rows, "queries": query_results,
              "candidates": [{"score": s, "a": location(chunks[a]), "b": location(chunks[b])}
                             for s, a, b in candidates[:100]]}
    dest = base / f".cache/{args.model}-report.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print(f"Report: {dest}")


if __name__ == "__main__":
    main()
