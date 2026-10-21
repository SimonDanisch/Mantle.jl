# Local semantic similarity search

Find related implementations and repeated responsibilities even when their
identifiers and control flow differ. This is review tooling, with no Mantle
runtime dependencies. Source stays local during inference; pretrained models
are downloaded from Hugging Face at pinned revisions.

## Run

Use the Julia environment in this directory to run `extract.jl`. It parses
`src/` and `ext/` without importing Mantle or evaluating macros. By default it
writes `.cache/chunks.toml`. From an existing Julia session:

```julia
using Pkg
Pkg.activate("tools/similarity")
Pkg.instantiate()
include("tools/similarity/extract.jl")
data = extract_chunks(pwd()) # repository root
mkpath("tools/similarity/.cache")
open(io -> TOML.print(io, data), "tools/similarity/.cache/chunks.toml", "w")
```

Create an isolated Python 3.12 environment, install an appropriate CUDA build
of PyTorch and `requirements.txt`, then run:

```sh
python tools/similarity/search.py --model qwen
python tools/similarity/search.py --model jina
python tools/similarity/search.py --model qwen --query 'GPU memory pressure and garbage collection rate limits'
```

The full run caches embeddings and emits a JSON report containing calibrated
code-to-code ranks, natural-language retrieval examples, and 100 cross-file
candidate pairs. Subsequent searches reuse the vectors. Changes to chunks,
model revision or source locations invalidate the cache. Model loading still
requires GPU memory; this implementation uses CUDA and BF16.
Calibration selectors use method signatures, so ordinary line movement does
not break the report. Removed methods are skipped explicitly. Results change
after consolidation; compare model quality on the same source snapshot.

## Method and limits

- JuliaSyntax supplies method boundaries, including short methods. Methods
  longer than 60 lines also produce overlapping 40-line regions, so shared
  substeps inside large functions can be found. Very small methods under 80
  characters, top-level declarations and non-method executable blocks are
  outside this initial index. This is not a complete duplication audit.
- Comments are removed with Julia's lexer while preserving source lines;
  strings and native shader source remain intact. File paths are metadata,
  not embedded text. Methods retain their names and type signatures.
- Qwen uses its documented last-token pooling and task instruction; Jina
  uses its code-to-code and natural-language-to-code prefixes. Complete
  methods are retained. Inputs exceeding 32,768 tokens produce an error
  instead of silently truncating.
- Search uses query embeddings against normalized document embeddings.
  The broad candidate list uses symmetric document-to-document cosine
  similarity. These are distinct retrieval modes; calibration evaluates
  the former. Region overlap can create several hits for one implementation.
- The lexical baseline is TF-IDF over identifiers and punctuation, without
  AST normalization. The six curated pairs are a small engineering check,
  not a held-out benchmark proving a universally best model.
- Related policy does not imply identical behavior. Review resource ownership,
  synchronization, error behavior, caller contracts and native requirements
  before choosing an abstraction. Scores are not equivalence probabilities.

The source index and model weights are deliberately excluded from commits.
Store reviewed findings and a dated report in `experiments/`, not the vector
cache. Rebuild after source changes before relying on file locations.

## Size accounting

`inventory.jl` uses the same parser and lexer to distinguish implementation
from docstrings, comments and blanks. A mixed code/comment line counts as code;
ordinary string literals, including embedded native shader source, stay code.
Docstrings are recognized by syntax, so `#` inside a shader or string is not
mistaken for a comment. These are source lines, not statement counts.

```julia
include("tools/similarity/inventory.jl")
write_inventory(pwd(), "tools/similarity/.cache/inventory.toml")
```

Use the same metric at both revisions. Moving documentation, reformatting code,
or moving implementations between files is not algorithm consolidation.

## Primary references

- [Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):
  general semantic and code retrieval; Apache-2.0 model license.
- [Jina Code Embeddings](https://huggingface.co/jinaai/jina-code-embeddings-0.5b):
  dedicated code-to-code retrieval; CC-BY-NC-4.0 model license.
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html):
  symmetric/asymmetric retrieval and direct search for small corpora.

Model licenses are separate from the license of this tooling.
