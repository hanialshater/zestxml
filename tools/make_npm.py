"""Build a larger real GZXML dataset: npm package descriptions tagged with keywords.

    python tools/make_npm.py GZXML-Datasets/GZ-NPM --cache /tmp/npm_cache.jsonl

This is the same shape of problem as the paper's GZ-Amazon: tag an item from a large,
long-tailed vocabulary of textual tags. Keywords make good labels for generalized
zero-shot XML because a keyword *is* text, so a label with no training example can still
be reached through the words it is made of.

Packages are collected from the public npm search API, which returns name, description
and keywords in bulk. The seed queries bias the sample topically -- worth remembering
when reading absolute numbers, though it affects any two implementations equally.
"""

import argparse
import json
import os
import random
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

SEARCH = "https://registry.npmjs.org/-/v1/search?text={q}&size={size}&from={frm}"
PAGES = 4
SIZE = 250
MIN_LABEL_FREQ = 10
UNSEEN_STRIDE = 4
UNSEEN_MIN_TEST = 8
TOKEN = re.compile(r"[a-z][a-z0-9]+")


def seeds():
    """A broad, deliberately mixed seed set: topic words plus neutral letter pairs."""
    words = """cli react vue angular http server client parser test build css html json xml
    aws azure google database sql mongo redis auth crypto stream file path date time math
    string array object util log debug config env docker kubernetes graphql rest api sdk
    lint format bundler webpack rollup babel typescript eslint jest mocha cypress svelte
    image video audio pdf csv excel markdown template render router state store cache queue
    email sms payment stripe oauth jwt session cookie validation schema orm migration
    websocket grpc proxy loadbalancer monitor metrics trace error retry mock fixture""".split()
    pairs = [a + b for a in string.ascii_lowercase for b in "aeiou"]
    return words + pairs


def fetch(cache_path, workers=8):
    if os.path.exists(cache_path):
        opener = __import__("gzip").open if cache_path.endswith(".gz") else open
        with opener(cache_path, "rt") as f:
            packages = {}
            for line in f:
                p = json.loads(line)
                packages[p["name"]] = p
        print(f"loaded {len(packages)} packages from {cache_path}")
        return packages

    jobs = [(q, page * SIZE) for q in seeds() for page in range(PAGES)]
    packages = {}

    def one(job):
        q, frm = job
        url = SEARCH.format(q=q, size=SIZE, frm=frm)
        try:
            with urlopen(Request(url, headers={"User-Agent": "zestxml-dataset-builder"}), timeout=60) as r:
                return json.load(r).get("objects", [])
        except Exception as exc:  # a failed page just means fewer packages
            print(f"  {q}@{frm}: {exc}", file=sys.stderr)
            return []

    with ThreadPoolExecutor(workers) as pool:
        for i, objects in enumerate(pool.map(one, jobs)):
            for o in objects:
                p = o.get("package", {})
                if p.get("name") and p.get("keywords"):
                    packages[p["name"]] = {
                        "name": p["name"],
                        "description": p.get("description", ""),
                        "keywords": p["keywords"],
                    }
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} queries, {len(packages)} packages with keywords")

    with open(cache_path, "w") as f:
        for p in packages.values():
            f.write(json.dumps(p) + "\n")
    print(f"fetched {len(packages)} packages -> {cache_path}")
    return packages


def build(out_dir, packages, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    docs = []
    for p in packages.values():
        name = " ".join(TOKEN.findall(p["name"].lower()))
        text = f"{name} {p['description']}".strip()
        tags = set()
        for kw in p["keywords"]:
            if not isinstance(kw, str):
                continue
            kw = kw.strip().lower()
            # a package tagged with its own name is trivially predictable, and not a tag
            if kw and kw != p["name"].lower() and len(kw) < 40:
                tags.add(kw)
        if text and tags:
            docs.append((text, sorted(tags)))

    freq = Counter(t for _, tags in docs for t in tags)
    labels = sorted(t for t, c in freq.items() if c >= MIN_LABEL_FREQ)
    label_id = {t: i for i, t in enumerate(labels)}
    docs = [(text, [t for t in tags if t in label_id]) for text, tags in docs]
    docs = [d for d in docs if d[1]]
    rng.shuffle(docs)

    split = int(0.75 * len(docs))
    trn, tst = docs[:split], docs[split:]

    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-z][a-z0-9]+", ngram_range=(1, 2),
        min_df=3, sublinear_tf=True, stop_words="english",
    )
    trn_X_Xf = vec.fit_transform([t for t, _ in trn])
    tst_X_Xf = vec.transform([t for t, _ in tst])
    Xf = list(vec.get_feature_names_out())
    xf_set = set(Xf)

    Yf, yf_id = [], {}

    def feat(name):
        if name not in yf_id:
            yf_id[name] = len(Yf)
            Yf.append(name)
        return yf_id[name]

    Y_Yf_rows, has_text = [], []
    for i, name in enumerate(labels):
        cells = {feat("__label__%d__%s" % (i, name)): 1.0}
        toks = TOKEN.findall(name)
        hit = False
        for t in toks:  # every token of the keyword becomes a label feature, matched or not
            cells[feat("1_" + t)] = 1.0
            hit = hit or t in xf_set
        phrase = " ".join(toks)
        if len(toks) > 1:
            cells[feat("1_" + phrase)] = 1.0
            hit = hit or phrase in xf_set
        Y_Yf_rows.append(sorted(cells.items()))
        has_text.append(hit)

    trn_X_Y = [sorted((label_id[t], 1.0) for t in tags) for _, tags in trn]
    tst_X_Y = [sorted((label_id[t], 1.0) for t in tags) for _, tags in tst]
    trn_freq = Counter(i for row in trn_X_Y for i, _ in row)
    tst_freq = Counter(i for row in tst_X_Y for i, _ in row)

    eligible = [
        i for i in range(len(labels))
        if tst_freq[i] >= UNSEEN_MIN_TEST and trn_freq[i] > 0 and has_text[i]
    ]
    eligible.sort(key=lambda i: -tst_freq[i])
    unseen = set(eligible[1::UNSEEN_STRIDE])
    unseen |= {i for i in range(len(labels)) if trn_freq[i] == 0}
    trn_X_Y = [[(i, v) for i, v in row if i not in unseen] for row in trn_X_Y]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.make_reuters import csr_rows, write_lines, write_smat

    # raw text, aligned row-for-row with the matrices below: needed by any encoder-based
    # baseline (Renee and friends read trn_X.txt / tst_X.txt / Y.txt)
    write_lines(f"{out_dir}/trn_X.txt", [t for t, _ in trn], len(trn))
    write_lines(f"{out_dir}/tst_X.txt", [t for t, _ in tst], len(tst))
    write_lines(f"{out_dir}/Y.txt", labels, len(labels))
    for fname in ("trn_filter_labels.txt", "tst_filter_labels.txt"):
        open(f"{out_dir}/{fname}", "w").close()  # no reciprocal pairs to filter here

    write_smat(f"{out_dir}/trn_X_Xf.txt", csr_rows(trn_X_Xf), len(Xf))
    write_smat(f"{out_dir}/tst_X_Xf.txt", csr_rows(tst_X_Xf), len(Xf))
    write_smat(f"{out_dir}/Y_Yf.txt", Y_Yf_rows, len(Yf))
    write_smat(f"{out_dir}/trn_X_Y.txt", trn_X_Y, len(labels))
    write_smat(f"{out_dir}/tst_X_Y.txt", tst_X_Y, len(labels))
    for name, vocab in (("Xf", Xf), ("Yf", Yf)):
        with open(f"{out_dir}/{name}.txt", "w") as f:
            f.write("\n".join(vocab) + "\n")
    with open(f"{out_dir}/unseen_labels.txt", "w") as f:
        for i in sorted(unseen):
            f.write("%d %s\n" % (i, labels[i]))

    unseen_mass = sum(tst_freq[i] for i in unseen)
    print(f"points      : {len(trn)} train / {len(tst)} test")
    print(f"labels      : {len(labels)} ({len(unseen)} unseen at train time)")
    print(f"features    : {len(Xf)} point / {len(Yf)} label")
    print(f"label text  : {sum(has_text)}/{len(labels)} keywords have a token in the point vocabulary")
    print(f"positives   : {sum(len(r) for r in trn_X_Y)} train / {sum(len(r) for r in tst_X_Y)} test "
          f"({sum(len(r) for r in tst_X_Y) / max(1, len(tst)):.2f} per test point)")
    print(f"unseen mass : {unseen_mass}/{sum(tst_freq.values())} test positives "
          f"({100.0 * unseen_mass / max(1, sum(tst_freq.values())):.1f}%) belong to unseen labels")
    print("wrote", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="GZXML-Datasets/GZ-NPM")
    ap.add_argument("--cache", default="data/npm_packages.jsonl.gz",
                    help="a committed snapshot; delete it to re-fetch from the registry")
    args = ap.parse_args()
    build(args.out_dir, fetch(args.cache))
