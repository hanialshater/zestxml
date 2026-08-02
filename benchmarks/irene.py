"""EMMETT / IRENE: synthesize a classifier for an unseen label from seen labels' classifiers.

    python benchmarks/irene.py GZXML-Datasets/GZ-NPM --model all-MiniLM-L6-v2

After Yadav et al., *Extreme Meta-Classification for Large-Scale Zero-Shot Retrieval*,
KDD 2024 (https://aka.ms/irene). This is a compact re-implementation of the idea, not of
the paper's system.

The idea inverts everything else in this repo. A per-label classifier is the strongest
scorer available, and `benchmarks/RESULTS.md` measures exactly how much stronger -- but it
cannot exist for a label with no training data, which is why one-vs-all scores *exactly
zero* on every unseen label here. The usual response, and every response tried in this
repo, is to describe the unseen label better: its tokens, its embedding, its semantic id.
IRENE instead builds the missing classifier out of the classifiers a label resembles:

    unseen "trail running"  ->  nearest SEEN labels: hiking, marathon, outdoor gear
                            ->  a generator combines their weight vectors
                            ->  a classifier for "trail running"

Three arms are measured, which is the point of the file:

``dual``      cosine between the document embedding and the label-name embedding. The
              Siamese baseline: it reaches unseen labels but is weak everywhere.
``ova``       one-vs-all classifiers on seen labels, nothing for unseen ones. Should score
              at the evaluator's tie-break floor on the unseen split, reproducing the
              result already recorded for OVA in RESULTS.md.
``mean``      classifiers for unseen labels as a similarity-weighted average of their
              neighbours'. The no-learning ablation, and the thing to beat before claiming
              the generator matters.
``irene``     the same neighbours through a learned single-layer attention generator,
              trained on *seen* labels leave-one-out so it never sees an unseen label.

Classifiers live in the encoder's embedding space rather than in sparse feature space --
that is what makes synthesis cheap, and it is what the paper does.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.eval import COLUMNS, evaluate, inv_propensity  # noqa: E402
from zestxml.io import read_desc_file, read_text_smat  # noqa: E402


def load(data_dir):
    names = read_desc_file(f"{data_dir}/Y.txt")
    trn_txt = read_desc_file(f"{data_dir}/trn_X.txt")
    tst_txt = read_desc_file(f"{data_dir}/tst_X.txt")
    trn_Y = read_text_smat(f"{data_dir}/trn_X_Y.txt")
    tst_Y = read_text_smat(f"{data_dir}/tst_X_Y.txt")
    unseen = torch.zeros(tst_Y.ncols, dtype=torch.bool)
    with open(f"{data_dir}/unseen_labels.txt") as f:
        for line in f:
            if line.strip():
                unseen[int(line.split()[0])] = True
    return names, trn_txt, tst_txt, trn_Y, tst_Y, unseen


def embed(texts, model, vectors, batch=256):
    if model:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model)
        v = enc.encode(list(texts), batch_size=batch, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
        return torch.as_tensor(v, dtype=torch.float32)
    from zestxml.embed import load_word_vectors
    from zestxml.quantize import embed_texts
    index, table = load_word_vectors(vectors, torch.float32)
    v, keep = embed_texts(texts, index, table, require_all=False)
    out = torch.zeros(len(texts), table.shape[1])
    out[torch.as_tensor(keep)] = torch.as_tensor(v)
    return out


class Generator(nn.Module):
    """One attention layer over the neighbour classifiers, as in IRENE's generator.

    The query is the novel label's own embedding, the keys and values are its neighbours'
    *classifier* vectors. Learning which neighbours to trust is the whole difference from
    the ``mean`` arm -- a fixed similarity weighting cannot tell a neighbour that shares a
    topic from one that shares a classifier.
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.dim = dim

    def forward(self, label_emb, nbr_clf):        # (B, d), (B, k, d)
        B, k, d = nbr_clf.shape
        h, hd = self.heads, d // self.heads
        q = self.q(label_emb).view(B, h, 1, hd)
        kk = self.k(nbr_clf).view(B, k, h, hd).transpose(1, 2)
        vv = self.v(nbr_clf).view(B, k, h, hd).transpose(1, 2)
        att = torch.softmax((q * kk).sum(-1) / hd ** 0.5, dim=-1)      # (B, h, k)
        mix = (att.unsqueeze(-1) * vv).sum(2).reshape(B, d)
        return self.norm(self.out(mix) + nbr_clf.mean(1))              # residual on the mean


def train_ova(X, Y, n_labels, seen_idx, epochs=15, lr=0.05, device="cpu"):
    """One-vs-all classifiers in embedding space, on the seen labels only."""
    W = torch.zeros(n_labels, X.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    target = torch.zeros(X.shape[0], n_labels, device=device)
    target[Y.row_ids(), Y.indices] = 1.0
    target = target[:, seen_idx]
    pos_w = ((target.numel() - target.sum()) / target.sum().clamp(min=1)).detach()
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0], device=device)
        tot = 0.0
        for lo in range(0, X.shape[0], 1024):
            b = perm[lo:lo + 1024]
            opt.zero_grad()
            loss = lossf(X[b] @ W[seen_idx].t(), target[b])
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 4:
            print(f"    ova epoch {ep+1}/{epochs} loss {tot:.4f}")
    return W.detach()


def neighbours(label_emb, seen_idx, k):
    """For every label, its k nearest SEEN labels (excluding itself)."""
    sims = label_emb @ label_emb[seen_idx].t()
    self_pos = torch.full((label_emb.shape[0],), -1, dtype=torch.long)
    self_pos[seen_idx] = torch.arange(len(seen_idx))
    rows = torch.arange(label_emb.shape[0])
    has_self = self_pos >= 0
    sims[rows[has_self], self_pos[has_self]] = -1e9        # leave-one-out
    val, idx = torch.topk(sims, k, dim=1)
    return seen_idx[idx], torch.softmax(val, dim=1)


def main(data_dir, model=None, vectors=None, k=16, epochs=15, gen_epochs=40, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names, trn_txt, tst_txt, trn_Y, tst_Y, unseen = load(data_dir)
    n_labels = tst_Y.ncols
    seen_idx = (~unseen).nonzero(as_tuple=True)[0]
    print(f"{len(trn_txt)} train / {len(tst_txt)} test, {n_labels} labels "
          f"({int(unseen.sum())} unseen), k={k}, device={device}")

    Xtr = embed(trn_txt, model, vectors).to(device)
    Xte = embed(tst_txt, model, vectors).to(device)
    L = embed(names, model, vectors).to(device)
    L = torch.nn.functional.normalize(L, dim=1)

    print("  training one-vs-all classifiers on the seen labels")
    W = train_ova(Xtr, trn_Y, n_labels, seen_idx.to(device), epochs=epochs, device=device)

    nbr_idx, nbr_w = neighbours(L.cpu(), seen_idx, k)
    nbr_idx, nbr_w = nbr_idx.to(device), nbr_w.to(device)

    # --- mean arm: similarity-weighted average of the neighbours' classifiers -----------
    W_mean = W.clone()
    W_mean[unseen] = (W[nbr_idx[unseen]] * nbr_w[unseen].unsqueeze(-1)).sum(1)

    # --- irene arm: learned generator, trained on SEEN labels leave-one-out -------------
    gen = Generator(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    tgt = torch.zeros(Xtr.shape[0], n_labels, device=device)
    tgt[trn_Y.row_ids().to(device), trn_Y.indices.to(device)] = 1.0
    print("  training the meta-classifier generator")
    for ep in range(gen_epochs):
        batch = seen_idx[torch.randperm(len(seen_idx))[:256]].to(device)
        synth = gen(L[batch], W[nbr_idx[batch]])
        # the synthesized classifier must work on real data, exactly as the true one does
        logits = Xtr @ synth.t()
        t = tgt[:, batch]
        pw = ((t.numel() - t.sum()) / t.sum().clamp(min=1)).detach()
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, t)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            print(f"    gen epoch {ep+1}/{gen_epochs} loss {loss.item():.4f}")

    with torch.no_grad():
        W_irene = W.clone()
        u = unseen.nonzero(as_tuple=True)[0].to(device)
        W_irene[u] = gen(L[u], W[nbr_idx[u]])

    # --- score and report ---------------------------------------------------------------
    truth = (tst_Y.to_dense() > 0).float()
    inv_prop = inv_propensity(trn_Y)
    arms = [("dual encoder (cosine)", Xte @ L.t()),
            ("one-vs-all", Xte @ W.t()),
            ("+ mean synthesis", Xte @ W_mean.t()),
            ("+ IRENE generator", Xte @ W_irene.t())]

    print("\n%-24s %7s %7s %9s %9s" % ("arm", "P@1", "PSP@5", "unseen P@1", "seen P@1"))
    for name, scores in arms:
        s = scores.detach().cpu().float()
        a = evaluate(s, truth, inv_prop)
        un = evaluate(s, truth, inv_prop, label_mask=unseen)
        se = evaluate(s, truth, inv_prop, label_mask=~unseen)
        print("%-24s %7.2f %7.2f %9.2f %9.2f" % (
            name, a["P@1"], a["PSP@5"], un["P@1"], se["P@1"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--model", help="sentence-transformers model")
    ap.add_argument("--vectors", help="GloVe file, if no --model")
    ap.add_argument("-k", type=int, default=16, help="neighbour classifiers per label")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--gen_epochs", type=int, default=40)
    a = ap.parse_args()
    main(a.data_dir, a.model, a.vectors, a.k, a.epochs, a.gen_epochs)
