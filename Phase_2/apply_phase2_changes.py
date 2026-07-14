#!/usr/bin/env python3
"""
apply_phase2_changes.py
=======================
Applies the five Phase-2 changes to the clean v9 pipeline
(airr_ml_submission_v9_parallel_task2.py).

Usage:
    python3 apply_phase2_changes.py INPUT.py OUTPUT.py

Each change is applied by a guarded string replacement. For every change the
script prints PATCHED / ALREADY PRESENT / **NOT FOUND**, so a silent no-op
(the failure mode that bit the original in-session patches) is impossible.

Changes:
  1. kmer5/6 disabled       -> NOT here; handled in run_phase2.sh via AIRR_MODELS
  2. SKIP_XGB=1  -> HAS_XGB=False
  3. Fast-scorer swap for slow (malidvj / pos_aa) Task 2 scorers
  4. REDUCE_TOPK=1 -> CFG.TOP_K_SEQUENCES=10000
  5. pos_aa empty-params KeyError fix (C default 1.0)
"""
import sys

def apply(content, name, old, new, *, expect=1):
    if new.strip() and new in content:
        print(f"  [{name}] ALREADY PRESENT - skipped")
        return content, True
    n = content.count(old)
    if n == 0:
        print(f"  [{name}] **NOT FOUND** - anchor missing, change NOT applied")
        return content, False
    if expect is not None and n != expect:
        print(f"  [{name}] **AMBIGUOUS** - anchor found {n}x (expected {expect}); NOT applied")
        return content, False
    print(f"  [{name}] PATCHED ({n} site{'s' if n != 1 else ''})")
    return content.replace(old, new), True


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python3 apply_phase2_changes.py INPUT.py OUTPUT.py")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        content = f.read()

    ok = True

    # -- Change 2: SKIP_XGB env var disables XGBoost --------------------------
    content, r = apply(
        content, "change 2 (SKIP_XGB)",
        "warnings.filterwarnings('ignore')",
        "# --- Phase 2 (change 2): disable XGBoost on large datasets via SKIP_XGB=1 ---\n"
        "if os.environ.get(\"SKIP_XGB\", \"0\") == \"1\":\n"
        "    HAS_XGB = False\n\n"
        "warnings.filterwarnings('ignore')",
    ); ok &= r

    # -- Change 4: REDUCE_TOPK env var lowers TOP_K ---------------------------
    content, r = apply(
        content, "change 4 (REDUCE_TOPK)",
        "CFG = Cfg()",
        "CFG = Cfg()\n"
        "# --- Phase 2 (change 4): reduce TOP_K on large datasets via REDUCE_TOPK=1 ---\n"
        "if os.environ.get(\"REDUCE_TOPK\", \"0\") == \"1\":\n"
        "    CFG.TOP_K_SEQUENCES = 10000",
    ); ok &= r

    # -- Change 5: pos_aa empty-best-params KeyError fix (C default 1.0) -------
    # bp can be {} when every C failed to beat AUC 0.5; bp['C'] then KeyErrors.
    # Applied to all C=bp['C'] sites: harmless where 'C' exists, fixes the crash
    # where it doesn't (the pos_aa / D95 case).
    content, r = apply(
        content, "change 5 (pos_aa C=1.0)",
        "C=bp['C']", "C=bp.get('C', 1.0)",
        expect=None,   # multiple legitimate sites
    ); ok &= r

    # -- Change 3: fast-scorer swap ------------------------------------------
    old_anchor = (
        "        info = self.model_info[selected_model]\n"
        "        mt = info.get('type', selected_model)"
    )
    new_block = (
        "        # --- Phase 2 (change 3): fast-scorer swap for slow Task 2 scorers ---\n"
        "        # MALIDVJ / pos_aa score sequentially (~5 min/repertoire). If a\n"
        "        # parallelisable model reached an equal (<=0.001) Task 1 AUC, use it\n"
        "        # for Task 2 instead; else fall back to the best fast model within 0.02.\n"
        "        _SLOW_SCORERS = {'malidvj', 'pos_aa'}\n"
        "        _FAST_SCOREABLE = {'kmer', 'emerson', 'kmer5_sgd', 'kmer6_sgd',\n"
        "                           'vj', 'vj_interact', 'vj_logfreq', 'vj_elasticnet'}\n"
        "        if (selected_model in self.model_info\n"
        "                and task2_method not in ('vj_positional', 'vj_ensemble')):\n"
        "            _sel_type = self.model_info[selected_model].get('type', selected_model)\n"
        "            if _sel_type in _SLOW_SCORERS:\n"
        "                _t1 = self.model.get('results', {}) if isinstance(self.model, dict) else {}\n"
        "                _sel_auc = _t1.get(selected_model)\n"
        "                if _sel_auc is not None:\n"
        "                    _fast = [(m, a) for m, a in _t1.items()\n"
        "                             if m in self.model_info\n"
        "                             and self.model_info[m].get('type', m) in _FAST_SCOREABLE]\n"
        "                    if _fast:\n"
        "                        _near = [(m, a) for m, a in _fast if abs(a - _sel_auc) <= 0.001]\n"
        "                        _pick = _near or [(m, a) for m, a in _fast if abs(a - _sel_auc) <= 0.02]\n"
        "                        if _pick:\n"
        "                            _swap = max(_pick, key=lambda x: x[1])\n"
        "                            if _swap[0] != selected_model:\n"
        "                                print(f\"  [Phase2 fast-scorer swap] {selected_model} \"\n"
        "                                      f\"(AUC {_sel_auc:.4f}) -> {_swap[0]} (AUC {_swap[1]:.4f}) \"\n"
        "                                      f\"for parallel Task 2 scoring\")\n"
        "                                selected_model = _swap[0]\n\n"
        "        info = self.model_info[selected_model]\n"
        "        mt = info.get('type', selected_model)"
    )
    content, r = apply(content, "change 3 (fast-scorer swap)", old_anchor, new_block); ok &= r

    with open(dst, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n" + ("ALL CHANGES APPLIED (or already present)." if ok
                  else "ONE OR MORE CHANGES FAILED - see **NOT FOUND** above."))
    print(f"Wrote: {dst}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
