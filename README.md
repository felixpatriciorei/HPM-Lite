# HPM-Lite

Small PyTorch testbed for controlled experiments on memory-augmented sequence models.

HPM-Lite tests whether simple external-memory mechanisms can improve exact long-range recall over a local-attention baseline. The project is intentionally small: synthetic tasks, CPU-friendly runs, explicit controls, and readable experiment outputs.

It is not a production model, benchmark leaderboard submission, or claim of a Transformer replacement.

## Purpose

Long-context models often mix several different problems under one label: recent-token lookup, compressed state tracking, exact retrieval of rare facts, and abstract reasoning. HPM-Lite isolates one narrow part of that problem:

> When a fact appears early in a sequence and the query appears after a long distractor gap, does a small memory mechanism help recover the exact answer?

The goal is not to prove a full architecture. The goal is to make the basic memory behavior measurable.

## What this repository demonstrates

* Python and PyTorch implementation
* Synthetic sequence-task generation
* Local-attention and memory-augmented baselines
* Exact answer-token evaluation
* Retrieval controls for leakage and shortcut checks
* CSV and Markdown reporting
* Small reproducible CPU experiments
* Basic tests and smoke runs

## Scope

HPM-Lite focuses on controlled recall diagnostics.

Included:

* local baseline
* recurrent summary baseline
* episodic memory model
* combined HPM-Lite model
* Hebbian-style associative-memory baseline
* validation controls
* structured memory diagnostics
* experiment reports

Not included:

* large-scale pretraining
* ANN-backed retrieval
* distributed training
* learned autonomous memory writing
* JEPA-style latent prediction
* full multi-path routing
* claims of general long-context reasoning

## Repository layout

```text
hpm_lite/        Core package
scripts/         Experiment and validation runners
tests/           Basic test coverage
docs/            Reports and result notes
requirements.txt Python dependencies
README.md        Project overview
```

## Install

```bash
git clone https://github.com/felixpatriciorei/HPM-Lite.git
cd HPM-Lite
pip install -r requirements.txt
```

The code does not require external datasets.

## Smoke test

Run a short CPU check:

```bash
python scripts/run_smoke.py
```

This trains the main model variants for a few steps and checks that losses, exact accuracy, and retrieval metrics run without NaNs.

## Train a model

```bash
python -m hpm_lite.train --model local --steps 200 --seq-len 512 --window 64 --batch-size 32
python -m hpm_lite.train --model epmem --steps 200 --seq-len 512 --window 64 --batch-size 32
python -m hpm_lite.train --model hpm_lite --steps 200 --seq-len 512 --window 64 --batch-size 32
```

Available model variants:

```text
local
recurrent
epmem
hpm_lite
hebbian
```

## Evaluate

```bash
python -m hpm_lite.evaluate \
  --checkpoint runs/path/to/checkpoint.pt \
  --model epmem \
  --seq-lens 256,512,1024 \
  --window 64
```

Evaluation reports exact answer accuracy and answer-position loss. If no checkpoint is supplied, the command evaluates a freshly initialized model, which is only useful for debugging.

## Run a small comparison

```bash
python scripts/run_experiment.py --steps 40 --seq-len 512 --window 64 --batch-size 16
```

The runner writes experiment outputs under `runs/` and produces summary files for inspection.

## Run validation controls

```bash
python scripts/run_validation.py --steps 5 --batch-size 4 --d-model 64 --layers 1 --device cpu
```

The validation runner checks whether memory results survive basic controls:

* normal memory
* shuffled memory values
* random memory keys
* no retrieval
* multiple seeds
* multiple sequence lengths

These controls are important because a model can appear to improve while actually relying on leakage, easy shortcuts, or task artifacts.

## Hebbian audit

```bash
python scripts/run_hebbian_audit.py --steps 2 --batch-size 4 --eval-batches 2 --d-model 64 --layers 1 --device cpu
```

This audit tests a simple associative-memory baseline under harder conditions such as two-hop recall, repeated keys, adjacent value IDs, corrupted values, random fact order, and distractor keys.

The update rule is:

```text
M = lambda * M + eta * k * v^T
r = M^T q
```

This baseline is useful because it separates “memory helped” from “the full model architecture was necessary.”

## Interpreting results

The main metric is exact answer accuracy at the answer position.

Answer cross-entropy is secondary. A small loss improvement is not enough if exact recall does not improve.

A memory model is only interesting when it beats the local baseline under long-gap conditions and survives retrieval controls. If the local baseline already solves the task, the task should be made harder by increasing the gap, increasing the number of facts, adding distractors, or using a multi-hop variant.

## Current research questions

This repository is organized around three practical questions:

1. Does episodic memory improve exact recall over a local baseline?
2. Do retrieval controls show that the model is using memory rather than shortcuts?
3. Do structured readouts recover facts when generic output heads fail?

Negative results are still useful. A failed memory mechanism is evidence about what not to scale.

## Known limitations

* Some experiments use oracle memory writes.
* Retrieval is brute-force rather than ANN-backed.
* Tasks are synthetic and controlled.
* Results should not be treated as general language-model performance.
* The code is designed for small experiments, not production training.
* The project does not implement the full HPM architecture.

## Suggested citation

```text
Patricio, F. P. J. HPM-Lite: controlled experiments for memory-augmented sequence recall. GitHub repository.
```

## Status

Experimental research code. Interfaces, scripts, and result formats may change.
