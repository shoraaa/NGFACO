
CXX      = g++-11
CXXFLAGS_COMMON = -std=c++17 -Wall -Wpedantic -Wextra -fexceptions -fopenmp

# Change to debug to compile with debugging flags
MODE = release

ifeq ($(MODE),release)
	CXXFLAGS = $(CXXFLAGS_COMMON) -O3 -g -march=native -flto -mavx2 -DNDEBUG
else
	CXXFLAGS = $(CXXFLAGS_COMMON) -g
endif

LDFLAGS  = 

TARGET = faco

BUILDDIR = build

SRCDIR = src

SOURCES = faco.cpp problem_instance.cpp local_search.cpp utils.cpp rand.cpp progargs.cpp

OBJS = $(SOURCES:.cpp=.o)

OUT_OBJS = $(addprefix $(BUILDDIR)/,$(OBJS))

# Python extension settings
PYTHON ?= python3
PYEXT_NAME = faco_cpp

.PHONY: clean all pyext pyext-install pyext-test pyext-clean

all: $(TARGET)

$(TARGET): $(OUT_OBJS)
	$(CXX) $(CXXFLAGS) $(OUT_OBJS) $(LDFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p results
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OUT_OBJS) $(TARGET)

# =============================================================================
# Python Extension Targets
# =============================================================================

# Build the Python extension in-place
pyext:
	cd $(SRCDIR) && $(PYTHON) setup.py build_ext --inplace

# Install the Python extension in development mode
pyext-install:
	cd $(SRCDIR) && pip install -e .

# Test the Python extension
pyext-test: pyext
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRCDIR)'); \
		import faco_cpp; print('faco_cpp imported successfully'); \
		import numpy as np; \
		n = 10; \
		dist = np.random.rand(n, n).astype(np.float32); \
		dist = (dist + dist.T) / 2; \
		np.fill_diagonal(dist, 0); \
		solver = faco_cpp.MFACO_TSP(dist, n_ants=5); \
		print(f'Created solver: n={solver.n}, k={solver.k}, best_cost={solver.best_cost:.2f}'); \
		costs, flats, _, _, _ = solver.sample(require_prob=False); \
		print(f'Sampled costs (fast): {costs}'); \
		costs2, flats2, _, _, traces = solver.sample(require_prob=True); \
		print(f'Sampled costs (traced): {costs2}'); \
		print(f'Trace n_decisions: {traces.n_decisions}'); \
		print('All tests passed!')"

# Clean Python extension build artifacts
pyext-clean:
	cd $(SRCDIR) && rm -rf build/ *.so *.egg-info/ __pycache__/
	rm -f $(SRCDIR)/faco_cpp*.so
