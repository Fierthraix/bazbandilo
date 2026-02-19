# bazbandilo

A chaotic communications simulator in Rust, with Python-based analysis and plotting.

## Docs

- API docs: <https://docs.rs/bazbandilo/latest/bazbandilo/>

## Tooling In Use (Current)

The active workflow is based on:

- Rust tests for BER/PD dataset generation
- `just` recipes from `justfile` for common tasks
- `uv` for Python environment + script execution
- `pitono/cfar.py` for the main figure set

Older plotting flows (`foo` / `goo`) were removed from the `justfile` and are not documented here.

## Prerequisites

- Rust: <https://rustup.rs/>
- Python 3.13+
- `uv`: <https://docs.astral.sh/uv/>
- `just`: <https://github.com/casey/just>

## Setup

```bash
uv sync --group dev
just make
```

`just make` runs `maturin develop` through `uv` to build/install the Python extension.

## Main Test Targets

The main data-generation tests are:

1. BER dataset

```bash
just ber
```

Writes BER progress/results to `/tmp/bers.json`.

2. PD dataset (main path used for figure generation)

```bash
just test pd_normal
```

Writes PD progress/results to `/tmp/results.json`.

Optional variant:

```bash
just test pd_inflated64
```

Also writes to `/tmp/results.json` (last run wins).

## Main Figure Pipeline

After generating `/tmp/bers.json` and `/tmp/results.json`, run:

```bash
just cfar -s -f 0.01 \
  -b /tmp/bers.json \
  -p /tmp/results.json \
  -d final/graphs/cfar_ssca
```

This uses `pitono/cfar.py` (via `just`) and writes the main figure set under `final/graphs/cfar_ssca/`, including:

- `cfar_<Detector>_pfa_0.01.png`
- `ber_<Modulation>_pfa_0.01.png`
- `lambda_<Detector>_pfa_0.01.png`
- `covert_metric_<Detector>_pfa_0.01.png`

If you want to use repository-level files instead of `/tmp` outputs, `cfar.py` defaults to:

- BER input: `./bers_curr.json`
- PD input: `./results_curr.json`
- Save directory: `/tmp/` (unless `-d` is set)

## Quick Reference

- List recipes:

```bash
just --list
```

- Run any specific test target:

```bash
just test <target-name>
```
