# make build      configure and build the C++ library, tests and examples
# make test       run the C++ tests
# make install    build and install the Python module (editable)
# make notebooks  execute every solution notebook top to bottom
# make exercises  regenerate the exercise notebooks from the solutions
# make figures    regenerate the figures in 1_theory/figures
# make format     clang-format the C++ sources
# make all        build, test, install, notebooks

BUILD_DIR ?= build/cmake
JOBS      ?= $(shell nproc 2>/dev/null || echo 2)
PYTHON    ?= python3
STYLE     := {BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 110, AccessModifierOffset: -2, DerivePointerAlignment: false, PointerAlignment: Left}

.PHONY: all build test install notebooks exercises figures format clean

all: build test install notebooks

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -j $(JOBS)

install:
	$(PYTHON) -m pip install -e ".[notebooks]"

notebooks:
	$(PYTHON) tools/run_notebooks.py 2_notebooks/solutions

exercises:
	$(PYTHON) tools/make_exercises.py

figures:
	$(PYTHON) tools/make_figures.py

format:
	clang-format -i -style='$(STYLE)' cpp/include/state_estimation/*.hpp cpp/src/*.cpp cpp/tests/*.cpp cpp/examples/*.cpp \
		python/state_estimation/bindings/*.cpp python/state_estimation/bindings/*.hpp

clean:
	rm -rf build
