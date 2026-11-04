# This conftest must stay import-free. pytest loads it before collecting the
# permission checks, so any third-party import here would reintroduce the full
# sglang dependency install that these checks are designed to avoid.
#
# Do not add `import sglang`, `import pytest`, or any other import to this file.
