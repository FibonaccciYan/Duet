# Legacy scripts

This directory contains historical experiments retained only for reproducibility.
They are not formal entry points and may contain fixed GPU assignments, proxy
settings, output directories, checkpoint locations, or experiment-specific
parameters.

The `quality_dense_*.sh` files reproduce earlier multi-GPU quality runs. Review
all environment variables before using them on another machine.

`llada_query_losa_correlation.py` and its wrapper are retained because the
correlation study is no longer part of the formal paper entry points, while its
analysis code and regression test remain useful historical evidence.
