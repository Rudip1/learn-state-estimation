# How to run

## Requirements

- A C++17 compiler (GCC ≥ 9 or Clang ≥ 10), CMake ≥ 3.20, git.
- Eigen 3.4 (`sudo apt install libeigen3-dev` on Ubuntu). If CMake does not find it, it downloads the headers.
- Python ≥ 3.10 with `pip`.
- Catch2 v3 is fetched by CMake when the tests are built.

## Build and test the C++

```bash
make build      # cmake -S . -B build/cmake && cmake --build build/cmake
make test       # ctest --test-dir build/cmake --output-on-failure
```

The examples are built next to the tests, one per chapter where it makes sense:

```bash
./build/cmake/cpp/examples/example_01_probability 0.5
```

## Install the Python module

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e ".[notebooks]"                           # builds the C++ and the pybind11 module
python -c "import state_estimation as se; print(se.chi2_quantile(0.95, 2))"
```

`pip install -e .` uses scikit-build-core: CMake is configured with `SKBUILD=ON`, which builds only the library
and the `_core` extension. Re-running the command rebuilds after C++ changes.

## Notebooks

```bash
jupyter lab 2_notebooks/exercises       # the reader's copy, with ✏️ cells to fill in
make notebooks                          # execute every solution notebook top to bottom (as CI does)
```

- `2_notebooks/solutions/` — complete notebooks; they run top to bottom and are executed in CI.
- `2_notebooks/exercises/` — generated from the solutions by `make exercises`, which replaces every block between
  `### BEGIN SOLUTION` and `### END SOLUTION` with a placeholder and clears all outputs. Edit the solutions, then
  regenerate; never edit the exercise copies by hand.

### Colab

Open a notebook from GitHub in Colab (`File → Open notebook → GitHub`, or replace `github.com` by
`colab.research.google.com/github` in the notebook URL). The first cell installs the module with
`%pip install git+https://github.com/Rudip1/learn-state-estimation`; compiling the C++ takes a few minutes.

## Figures

Every figure in `1_theory/figures/` is produced by `tools/make_figures.py` from the library:

```bash
make figures                 # all chapters
python tools/make_figures.py 05   # one chapter
```

## Formatting

```bash
make format                  # clang-format, Google base style, 4-space indent, 110 columns
```
