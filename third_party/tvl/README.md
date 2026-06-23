# TVL baseline code

This directory contains TVL-derived code vendored for baseline reproducibility.
It is kept separate from the SPLASH implementation under `src/` so the project
boundary between SPLASH code and third-party baseline code is explicit.

Before public release, verify the original TVL source URL, license, citation,
and any local modifications, then record them here.

SPLASH runtime code should use `src/util/tactile_preprocess.py` for tactile
image preprocessing instead of importing this package directly.
