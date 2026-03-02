# 2026-PCCP-ORBT-SI

## Setup
This repository requires **pyscf** : https://github.com/pyscf/ (version : 2.11.0)
This repository requires **pyscf-forge** : https://github.com/pyscf/pyscf-forge (for sfnoci)

## Installation Notes

This repo includes a modified `gbci` module. To use it, copy it into your pyscf-forge installation so it is imported from `pyscf.sfnoci`:

- Copy this repo’s `gbci.py` to: `<pyscf-forge>/pyscf/sfnoci/gbci.py`
