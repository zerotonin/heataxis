# heataxis

[![tests](https://github.com/zerotonin/heataxis/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/heataxis/actions/workflows/tests.yml)
[![docs](https://github.com/zerotonin/heataxis/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/heataxis/)
[![PyPI](https://img.shields.io/pypi/v/heataxis.svg)](https://pypi.org/project/heataxis/)
[![Python](https://img.shields.io/pypi/pyversions/heataxis.svg)](https://pypi.org/project/heataxis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Thermal heat-load indices and physiological threshold detection for dairy-cattle
heat-stress analysis.**

`heataxis` is one tool covering the full pipeline behind a two-part companion
paper series:

- **`heataxis.indices`** — *the x-axis.* Cattle thermal / heat-load indices
  (THI variants, BGHI, ETI, THI_adj, HLI, ETIC, ...), grouped by heat-exchange
  pathway. *(Paper I: Detecting heat stress in dairy cattle — Choosing the
  thermal index.)*
- **`heataxis.thresholds`** — *the y-axis.* Methods that locate the
  environmental threshold at which a physiological response changes
  (broken-stick, Hill/4PL, derivative exceedance, ...). *(Paper II — Locating
  the physiological threshold.)*

Pick an index, then detect the threshold — in a single install.

## Installation

```bash
pip install heataxis            # core (indices)
pip install "heataxis[stats]"   # + scipy-based threshold methods
pip install "heataxis[viz]"     # + matplotlib plotting helpers
```

## Quickstart

```python
import numpy as np
from heataxis import indices, thresholds

# x-axis: compute a thermal index from barn climate
thi = indices.thi_nrc(ta=28.0, rh=60.0)          # 77.03

# y-axis: locate the threshold in a response vs index relationship
x = np.linspace(50, 80, 200)
y = np.where(x < 68, 38.5, 38.5 + 0.05 * (x - 68))
fit = thresholds.broken_stick(x, y)
print(fit.threshold)                              # ~68
```

## Documentation

Full API reference and the index catalogue: https://zerotonin.github.io/heataxis/

## Citing

See [`CITATION.cff`](CITATION.cff). A release DOI will be minted via Zenodo.

## License

MIT — see [`LICENSE`](LICENSE).
