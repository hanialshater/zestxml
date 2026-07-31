"""better_vectors: does a better embedding space rescue the fuzzy direct map?

Stage A (prep): read the big .gz word-vector tables ONCE, filter to the tokens that
GZ-NPM / GZ-Reuters actually need, and write small plain word2vec text files that the
pipeline can re-read cheaply for every configuration.

ConceptNet Numberbatch keys look like ``/c/en/web_server``; we keep only the ``/c/en/``
rows and strip the prefix (underscores -> we index the whole underscored token AND, as a
convenience, nothing else -- zestxml.embed tokenises on spaces/hyphens so single tokens
are what matter).

Stage B (report): per-dataset token coverage of the label-feature vocabulary and of the
point-feature vocabulary, which is the evidence for the coverage-vs-something-else
question.
"""

from __future__ import annotations

import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.io import read_desc_file  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user/305e430a-e06b-53e3-bf8b-d5b189305458/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {
    "glove": os.path.join(SCRATCH, "glove100.gz"),
    "fasttext": os.path.join(SCRATCH, "fasttext-wiki-news-subwords-300.gz"),
    "numberbatch": os.path.join(SCRATCH, "conceptnet-numberbatch-17-06-300.gz"),
}
DATASETS = {
    "npm": os.path.join(REPO, "GZXML-Datasets", "GZ-NPM"),
    "reuters": os.path.join(REPO, "GZXML-Datasets", "GZ-Reuters-90"),
}


def tokens(name: str):
    return name.lower().replace("-", " ").split()


def query_name(name: str) -> str:
    return name[name.find("_") + 1:]


def is_token_feature(q: str) -> bool:
    return bool(q) and not q.startswith("_") and "__" not in q


def dataset_vocab(d: str):
    """(point-feature tokens, token-like label-feature tokens) for one dataset."""
    Xf = read_desc_file(os.path.join(d, "Xf.txt"))
    Yf = read_desc_file(os.path.join(d, "Yf.txt"))
    xf_toks = {t for n in Xf for t in tokens(n)}
    queries = [query_name(n) for n in Yf]
    yf_toks = {t for q in queries if is_token_feature(q) for t in tokens(q)}
    return xf_toks, yf_toks


def all_needed():
    need = set()
    for d in DATASETS.values():
        a, b = dataset_vocab(d)
        need |= a | b
    return need


def prep(key: str, out_path: str):
    need = all_needed()
    src = SOURCES[key]
    kept = {}
    opener = gzip.open if src.endswith(".gz") else open
    with opener(src, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sp = line.find(" ")
            if sp <= 0:
                continue
            w = line[:sp]
            if key == "numberbatch":
                if not w.startswith("/c/en/"):
                    continue
                w = w[6:]
            if w in need and w not in kept:
                kept[w] = line[sp + 1:].strip()
    dim = len(next(iter(kept.values())).split())
    with open(out_path, "w") as f:
        f.write("%d %d\n" % (len(kept), dim))
        for w, v in kept.items():
            f.write("%s %s\n" % (w, v))
    print("%s: kept %d/%d needed tokens, dim %d -> %s" % (key, len(kept), len(need), dim, out_path))


def coverage(vec_path: str):
    have = set()
    with open(vec_path) as f:
        f.readline()
        for line in f:
            have.add(line[:line.find(" ")])
    for name, d in DATASETS.items():
        xf_toks, yf_toks = dataset_vocab(d)
        print("  %-8s label-feature tokens %5d covered %5.1f%% | point-feature tokens %6d covered %5.1f%%" % (
            name, len(yf_toks), 100.0 * len(yf_toks & have) / max(1, len(yf_toks)),
            len(xf_toks), 100.0 * len(xf_toks & have) / max(1, len(xf_toks))))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif cmd == "coverage":
        for p in sys.argv[2:]:
            print(os.path.basename(p))
            coverage(p)
