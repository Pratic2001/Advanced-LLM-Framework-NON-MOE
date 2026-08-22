"""Regression test for the train_grpo.py CLI / train loop attribute mismatch.

Background: train_grpo.py exposed `--num-steps` on the CLI but its training
loop referenced `args.max_steps`. Any run that passed `--num-steps` failed
immediately with `AttributeError: 'Namespace' object has no attribute
'max_steps'. Did you mean: 'num_steps'?`.

This test asserts the source no longer references `args.max_steps` anywhere
(non-comment), so the regression can't quietly come back.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_GRPO = os.path.join(REPO_ROOT, "train_grpo.py")


def test_train_grpo_does_not_reference_args_max_steps():
    with open(TRAIN_GRPO) as f:
        src = f.read()

    bad = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if "args.max_steps" in line and not line.lstrip().startswith("#"):
            bad.append((lineno, line))
    assert not bad, (
        "train_grpo.py still references args.max_steps; the training loop "
        "would AttributeError on every run. Offending lines:\n"
        + "\n".join(f"  {n}: {l}" for n, l in bad)
    )


def test_train_grpo_cli_exposes_num_steps():
    with open(TRAIN_GRPO) as f:
        src = f.read()
    # The CLI must register `--num-steps`. argparse converts dashes to
    # underscores in the dest, so the attribute name is `args.num_steps`.
    assert "--num-steps" in src, \
        "train_grpo.py must expose --num-steps on argparse"
    assert "args.num_steps" in src, \
        "train_grpo.py training loop must read args.num_steps"