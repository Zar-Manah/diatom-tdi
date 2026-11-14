#!/usr/bin/env python3
"""Fase 0 MAX local — split 5-fold sellado a partir de split_f1 + label_audit.

- pool = train+val de split_f1 MENOS los suspects (cuarentena train+val).
- test = TEST SELLADO de split_f1, idéntico byte a byte (comparabilidad).
- 5 folds estratificados (seed 4242): folds[k] = {train, val}.
- class_weights desde el audit (solo errores de train).
- hash SHA256 sobre (classes, test, folds, seed, pool_excluded) sellado.
Salida: state/split_max.json
"""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("data 🧬")
SPLIT_F1 = BASE / "state/split_f1.json"
AUDIT = BASE / "state/label_audit.json"
OUT = BASE / "state/split_max.json"
SEED = 4242
N_FOLDS = 5


def main() -> int:
    split = json.loads(SPLIT_F1.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit.get("split_hash") != split["hash"]:
        print("FATAL: audit de un split distinto")
        return 2

    suspects = {s["rel"] for s in audit["suspects"]}
    val_suspects = {s["rel"] for s in audit["val_suspects"]}
    excluded = suspects | val_suspects
    print(f"suspects train={len(suspects)} val={len(val_suspects)} "
          f"(excluidos del pool)")

    pool = [(r, y) for r, y in split["train"] + split["val"] if r not in excluded]
    test = split["test"]
    classes = split["classes"]

    by_class = defaultdict(list)
    for r, y in pool:
        by_class[y].append((r, y))
    rng = random.Random(SEED)
    folds = [dict(train=[], val=[]) for _ in range(N_FOLDS)]
    for y in sorted(by_class):
        items = by_class[y]
        rng.shuffle(items)
        for i, it in enumerate(items):
            folds[i % N_FOLDS]["val"].append(it)
    for k in range(N_FOLDS):
        val_set = {r for r, _ in folds[k]["val"]}
        folds[k]["train"] = [(r, y) for r, y in pool if r not in val_set]

    for k, f in enumerate(folds):
        assert not (set(r for r, _ in f["train"]) & set(r for r, _ in f["val"]))
        print(f"fold{k}: train={len(f['train'])} val={len(f['val'])}")

    split_out = dict(
        classes=classes,
        genus=split["genus"],
        test=test,
        folds=folds,
        seed=SEED,
        n_folds=N_FOLDS,
        pool_size=len(pool),
        excluded=list(sorted(excluded)),
        class_weights=audit["class_weights"],
        audit_summary=dict(
            n_suspects=audit["n_suspects"],
            n_val_suspects=audit["n_val_suspects"],
            prob_min=audit["prob_min"],
            f2_best_acc=audit["best_acc"],
        ),
        images_prefix=split["images_prefix"],
        source=split["source"],
        parent_split_hash=split["hash"],
    )
    blob = json.dumps(
        {k: split_out[k] for k in
         ("classes", "test", "folds", "seed", "n_folds",
          "images_prefix", "parent_split_hash")},
        sort_keys=True,
    )
    split_out["hash"] = hashlib.sha256(blob.encode()).hexdigest()

    OUT.write_text(json.dumps(split_out))
    print(f"test idéntico al sellado: {test == split['test']}")
    print(f"pool={len(pool)} excluded={len(excluded)} "
          f"hash={split_out['hash'][:16]}")
    print(f"saved -> {OUT}")

    st = json.loads((BASE / "state/state.json").read_text())
    st.update(
        phase="MAX",
        status="split_max_ready",
        split_max_hash=split_out["hash"],
        split_max_pool=len(pool),
        split_max_excluded=len(excluded),
        updated="2026-09-24",
    )
    (BASE / "state/state.json").write_text(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
