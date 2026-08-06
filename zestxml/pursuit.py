"""Structured hard pursuit over the bilinear pattern, keyed by semantic-ID prefixes.

    from zestxml.pursuit import sid_block_index, StructuredPursuit

    x_codes, y_codes = code_features(Xf, Yf, "glove.6B.100d.txt")
    blocks = sid_block_index(pattern, Yf, y_codes, x_codes)
    pursuit = StructuredPursuit(blocks, budget=0.5, interval=4)
    clf.fit(..., pursuit=pursuit)

The pattern that ZestXML mines is a set of ``(xf, yf)`` pairs and the model's only free
parameters are one weight per pair.  ``pattern.prune_by_similarity`` already showed that
*removing* mined pairs is worth doing -- on GZ-NPM it is the largest accuracy gain measured
in this repo (P@1 73.02 -> 74.57).  It has two limitations this module is built to fix.

**It prunes entries independently, on a proxy.**  An entry survives if the cosine between
its two feature names clears a floor.  Cosine is not the objective, and 45% of entries have
a name that cannot be embedded at all and pass through unjudged.  Here the score is the
weight the model actually learned, and every entry is judgeable because every feature falls
into some quantization cell.

**It prunes once, before training.**  A pair discarded before the first epoch never gets a
chance to prove itself.  This runs every ``interval`` epochs on the live weights, and blocks
can come back -- the gradient is accumulated *densely*, including for masked entries, so an
inactive block still has an importance signal.  That is the one detail that makes revival
possible at all; masking the parameter would zero its gradient and freeze it out forever.

**Blocks, not entries.**  A block at level ``l`` is every pattern entry whose two feature
names share the same pair of ``l``-length semantic-ID prefixes.  Dropping one drops a whole
coherent group of ``(document region, label region)`` links together rather than picking off
members of it, and
*ancestor closure* -- a block can be active only if its parent is -- means the surviving set
is a forest of complete paths rather than a scattering of deep cells whose context has been
pruned away.

Two feature kinds are held out of the mechanism:

* ``__label__<i>__<name>`` features are a label's private identity feature.  They are the
  per-item residual: the part of a label's classifier that is not shared with anything.
  They are never pruned, so a pruned model keeps exactly the capacity an unseen label
  cannot use anyway.
* features with no code (out of the embedding vocabulary) are collected into one reserved
  cell per level and are prunable as a group, which keeps the partition total.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# block construction
# --------------------------------------------------------------------------- #
@dataclass
class BlockIndex:
    """Which block each weight slot belongs to, at each level, plus the parent links.

    ``block_of[l][p]`` is the level-``l`` block of weight slot ``p``, numbered densely.
    ``parent[l][b]`` is the level-``l-1`` block containing level-``l`` block ``b``; level 0
    has no parents.  ``always[p]`` marks slots exempt from pruning.
    """

    block_of: List[Tensor]
    parent: List[Tensor]
    always: Tensor
    n_slots: int
    coded_frac: float = 0.0

    @property
    def levels(self) -> int:
        return len(self.block_of)

    def counts(self) -> List[int]:
        return [int(b.max()) + 1 if b.numel() else 0 for b in self.block_of]


def _dense_ids(keys: Tensor) -> Tuple[Tensor, Tensor]:
    """Renumber arbitrary integer keys to 0..n-1.  Returns (ids, first_key_per_id)."""
    uniq, inv = torch.unique(keys, return_inverse=True)
    return inv, uniq


def slot_features(pattern) -> Tuple[Tensor, Tensor]:
    """The ``(xf, yf)`` pair of every weight slot, indexed by slot rather than by row.

    ``BilinearPattern`` stores entries grouped by ``xf`` with ``pid`` pointing at the weight
    vector, and the weight vector is in a different order (it interleaves the two mined
    matrices).  Everything here is per-slot, so scatter the pairs into slot order once.
    """
    device = pattern.yf.device
    counts = pattern.row_counts()
    rows = torch.repeat_interleave(torch.arange(pattern.n_xf, device=device), counts)
    xf = torch.zeros(pattern.size, dtype=torch.long, device=device)
    yf = torch.zeros(pattern.size, dtype=torch.long, device=device)
    xf[pattern.pid] = rows
    yf[pattern.pid] = pattern.yf
    return xf, yf


def sid_block_index(
    pattern,
    Yf: Sequence[str],
    y_codes: np.ndarray,
    x_codes: Optional[np.ndarray] = None,
    levels: Optional[int] = None,
    side: str = "xy",
    keep_identity: bool = True,
) -> BlockIndex:
    """Group the pattern's weight slots by semantic-ID prefix.

    ``y_codes`` is ``(len(Yf), L)`` and ``x_codes`` ``(n_xf, L)``; a row of ``-1`` means
    the feature could not be coded.  ``side`` picks what a block is:

    ``"xy"``  ``(x-prefix, y-prefix)`` -- both sides quantized.  This is the only setting
              where blocks are genuinely coarse at every level.
    ``"y"``   the label-side prefix alone.  Very blunt: dropping a level-0 block removes a
              region of label space for *every* document word at once.
    ``"xf_y"``  the raw ``xf`` paired with the y-prefix.  Kept because it is the obvious
              first thing to try and it is worth being able to show why it fails: with
              35215 distinct ``xf`` over 284540 slots it produces 257634 level-0 blocks,
              1.1 slots each, so "prune a block" and "prune an entry" are the same
              operation and the structure does no work.

    ``keep_identity`` exempts ``__label__`` features from pruning.  They are the per-label
    residual, and on GZ-Reuters-90 they are 45% of all slots -- so with it on, a budget of
    0.5 keeps ~72% of the weight vector.  The budget is always a fraction of the *prunable*
    slots; :meth:`StructuredPursuit.step` logs both numbers.
    """
    device = pattern.yf.device
    xf, yf = slot_features(pattern)
    L = int(y_codes.shape[1]) if levels is None else min(levels, int(y_codes.shape[1]))
    K = max(int(y_codes.max()) if y_codes.size else 0,
            int(x_codes.max()) if x_codes is not None and x_codes.size else 0) + 1

    is_identity = torch.tensor(
        [n.startswith("__label__") for n in Yf], dtype=torch.bool, device=device)
    always = is_identity[yf] if keep_identity else torch.zeros(
        pattern.size, dtype=torch.bool, device=device)

    def as_cells(codes):
        # one reserved cell per level for the uncoded, so the partition stays total
        t = torch.as_tensor(np.ascontiguousarray(codes[:, :L]), device=device)
        return torch.where(t >= 0, t, torch.full_like(t, K)), (t >= 0).all(1)

    yc, y_ok = as_cells(y_codes)
    if side == "xy":
        if x_codes is None:
            raise ValueError("side='xy' needs x_codes; pass side='y' to code labels only")
        xc, _ = as_cells(x_codes)

    block_of: List[Tensor] = []
    parent: List[Tensor] = []
    prefix = torch.zeros(pattern.size, dtype=torch.long, device=device)
    prev_ids: Optional[Tensor] = None
    for l in range(L):
        if side == "xy":
            prefix = (prefix * (K + 1) + xc[xf, l]) * (K + 1) + yc[yf, l]
            key = prefix
        elif side == "y":
            prefix = prefix * (K + 1) + yc[yf, l]
            key = prefix
        elif side == "xf_y":
            prefix = prefix * (K + 1) + yc[yf, l]
            key = xf * ((K + 1) ** (l + 1)) + prefix
        else:
            raise ValueError(f"unknown side {side!r}")
        ids, _ = _dense_ids(key)
        block_of.append(ids)
        n = int(ids.max()) + 1 if ids.numel() else 0
        p = torch.zeros(n, dtype=torch.long, device=device)
        if prev_ids is not None:
            # every slot in a level-l block shares one level-(l-1) block by construction,
            # so scattering the parent of any member is enough
            p[ids] = prev_ids
        parent.append(p)
        prev_ids = ids

    return BlockIndex(block_of, parent, always, pattern.size,
                      coded_frac=float(y_ok[yf].float().mean()) if pattern.size else 0.0)


# --------------------------------------------------------------------------- #
# the pursuit itself
# --------------------------------------------------------------------------- #
def _block_norm(values: Tensor, block_of: Tensor, n_blocks: int) -> Tensor:
    out = torch.zeros(n_blocks, dtype=values.dtype, device=values.device)
    out.index_add_(0, block_of, values * values)
    return out.sqrt()


class StructuredPursuit:
    """Iterative block masking with ancestor closure and gradient-driven revival.

    ``budget`` is the fraction of *prunable* slots to keep (or an absolute count above 1).
    ``level_budgets`` optionally caps the number of active blocks at each level, applied
    coarse-to-fine before the slot budget.  ``explore`` is the drop-and-grow rate: that
    fraction of the active set is swapped each round, decayed linearly to zero so the mask
    settles before training ends.

    Selection follows RigL: among currently active blocks the lowest weight norm is dropped,
    among inactive ones the highest accumulated-gradient norm is grown.  The two are never
    compared on one scale -- they are separate rankings over disjoint sets, which is what
    lets a magnitude score and a gradient score coexist without a calibration constant.
    """

    def __init__(
        self,
        blocks: BlockIndex,
        budget: float = 0.5,
        interval: int = 4,
        explore: float = 0.3,
        level_budgets: Optional[Sequence[float]] = None,
        log=print,
    ):
        self.blocks = blocks
        self.budget = budget
        self.interval = max(1, int(interval))
        self.explore = explore
        self.level_budgets = list(level_budgets) if level_budgets else None
        self.log = log
        self.rounds = 0
        device = blocks.always.device
        self.mask = torch.ones(blocks.n_slots, device=device)
        self.grad_sq = torch.zeros(blocks.n_slots, device=device)
        self._grad_steps = 0

    # ---------------------------------------------------------------- gradient
    def accumulate(self, grad: Tensor) -> None:
        """Take the *dense* gradient of the effective weights, masked entries included."""
        self.grad_sq += grad.detach().float() ** 2
        self._grad_steps += 1

    def hook(self, effective: Tensor) -> Tensor:
        """Attach the accumulator to the masked weight tensor and return it unchanged."""
        if effective.requires_grad:
            effective.register_hook(lambda g: (self.accumulate(g), g)[1])
        return effective

    # ------------------------------------------------------------------ budget
    def _as_count(self, spec: float, total: int) -> int:
        return int(round(spec)) if spec > 1 else max(1, int(round(spec * total)))

    def _rate(self, epoch: int, epochs: int) -> float:
        return self.explore * 0.5 * (1.0 + np.cos(np.pi * min(1.0, epoch / max(1, epochs))))

    # -------------------------------------------------------------------- step
    def step(self, weights: Tensor, epoch: int = 0, epochs: int = 1) -> None:
        b = self.blocks
        if b.levels == 0:
            return
        prunable = ~b.always
        w = (weights.detach() * self.mask).float()
        g = self.grad_sq.sqrt()
        rate = self._rate(epoch, epochs)

        # active[l] is over level-l blocks; walk coarse to fine so closure can be applied
        # as we go and a dropped parent removes its whole subtree before we score inside it
        active: List[Tensor] = []
        for l in range(b.levels):
            bo, n = b.block_of[l], b.counts()[l]
            alive = _block_norm(torch.where(prunable, self.mask, torch.zeros_like(self.mask)),
                                bo, n) > 0
            w_norm = _block_norm(torch.where(prunable, w, torch.zeros_like(w)), bo, n)
            g_norm = _block_norm(torch.where(prunable, g, torch.zeros_like(g)), bo, n)
            # a block containing only exempt slots must stay: pruning it would mask them
            exempt_only = _block_norm(b.always.float(), bo, n) > 0
            exempt_only &= ~(_block_norm(prunable.float(), bo, n) > 0)

            keep = alive.clone()
            if self.level_budgets is not None and l < len(self.level_budgets):
                cap = self._as_count(self.level_budgets[l], n)
                if cap < int(keep.sum()):
                    order = torch.argsort(w_norm, descending=True)
                    sel = torch.zeros_like(keep)
                    sel[order[:cap]] = True
                    keep &= sel
            n_active = max(1, int(keep.sum()))
            n_drop = int(round(rate * n_active))
            if n_drop > 0:
                cand = keep.nonzero(as_tuple=True)[0]
                worst = cand[torch.argsort(w_norm[cand])[:n_drop]]
                keep[worst] = False
                dead = (~keep).nonzero(as_tuple=True)[0]
                if dead.numel():
                    best = dead[torch.argsort(g_norm[dead], descending=True)[:n_drop]]
                    keep[best] = True
            keep |= exempt_only
            if l > 0:  # ancestor closure
                keep &= active[l - 1][b.parent[l]]
            active.append(keep)

        finest = active[-1][b.block_of[-1]]
        new_mask = torch.where(b.always, torch.ones_like(self.mask), finest.to(self.mask.dtype))

        # the slot budget is the last word, and it is applied within the surviving forest:
        # rank the still-active prunable slots by weight and cut the tail.
        # Slots in a block that was just revived are held out of that cut for one round.
        # They necessarily have weight zero -- the mask forced it -- so ranking them by
        # magnitude would re-kill every one of them immediately and revival, which is the
        # whole reason the gradient is accumulated densely, could never actually happen.
        revived_mask = (new_mask > 0) & (self.mask == 0)
        cap = self._as_count(self.budget, int(prunable.sum()))
        live = (new_mask > 0) & prunable & ~revived_mask
        if int(live.sum()) > cap:
            idx = live.nonzero(as_tuple=True)[0]
            drop = idx[torch.argsort(w[idx].abs())[: int(live.sum()) - cap]]
            new_mask[drop] = 0.0

        revived = int(revived_mask.sum())
        self.mask = new_mask
        weights.data.mul_(self.mask)
        self.grad_sq.zero_()
        self._grad_steps = 0
        self.rounds += 1
        if self.log:
            self.log("  pursuit round %d: %d/%d slots active (%.1f%%), blocks %s, %d revived"
                     % (self.rounds, int(self.mask.sum()), self.mask.numel(),
                        100.0 * float(self.mask.mean()),
                        "/".join(str(int(a.sum())) for a in active), revived))

    # ------------------------------------------------------------------ report
    def stats(self) -> dict:
        b = self.blocks
        out = {"active_slots": int(self.mask.sum()), "total_slots": self.mask.numel()}
        for l in range(b.levels):
            n = b.counts()[l]
            live = _block_norm(self.mask, b.block_of[l], n) > 0
            out[f"active_blocks_l{l}"] = int(live.sum())
            out[f"total_blocks_l{l}"] = n
        return out


# --------------------------------------------------------------------------- #
# codes for label features
# --------------------------------------------------------------------------- #
def code_features(
    Xf: Sequence[str],
    Yf: Sequence[str],
    vectors: str,
    levels: int = 3,
    codebook_size: int = 64,
    seed: int = 0,
    log=print,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize point-feature and label-feature *names* into semantic-ID paths.

    Returns ``(x_codes, y_codes)``, shapes ``(len(Xf), levels)`` and ``(len(Yf), levels)``,
    with ``-1`` rows for names that could not be embedded.

    One codebook is fit over *both* vocabularies together, not one each.  A block is a pair
    of prefixes, so the two sides have to be quantized against the same centroids for
    "``xf`` and ``yf`` landed in the same region" to mean anything.  Fitting separately
    would give two unrelated numberings and the blocks would be arbitrary.

    ``__label__`` identity features are left uncoded on purpose: they name a specific label
    rather than a concept, and they are exempt from pruning anyway.
    """
    from .quantize import embed_texts, rq_kmeans, sharing_report

    y_names = [n[n.find("_") + 1:] for n in Yf]  # "1_hiking" is compared through "hiking"
    y_ok = [i for i, n in enumerate(Yf) if not n.startswith("__label__")]
    texts = list(Xf) + [y_names[i] for i in y_ok]

    if os.path.exists(vectors):
        from .embed import load_word_vectors
        index, table = load_word_vectors(
            vectors, torch.float32, vocab={t for n in texts for t in n.split()})
        emb, keep = embed_texts(texts, index, table.numpy(), require_all=True)
    else:
        from .quantize import encode_texts
        emb, keep = encode_texts(texts, model=vectors, log=log)

    x_out = np.full((len(Xf), levels), -1, dtype=np.int64)
    y_out = np.full((len(Yf), levels), -1, dtype=np.int64)
    if emb.shape[0] == 0:
        log("pursuit: nothing could be embedded -- every block is the reserved cell")
        return x_out, y_out

    _, codes = rq_kmeans(emb, levels=levels, codebook_size=codebook_size, seed=seed, log=log)
    y_index = np.asarray(y_ok, dtype=np.int64)
    for row, src in zip(keep, codes):
        if row < len(Xf):
            x_out[row] = src
        else:
            y_out[y_index[row - len(Xf)]] = src
    log("pursuit: coded %d/%d point features and %d/%d label features"
        % (int((x_out[:, 0] >= 0).sum()), len(Xf),
           int((y_out[:, 0] >= 0).sum()), len(Yf)))
    for line in sharing_report(codes):
        log("  " + line)
    return x_out, y_out
