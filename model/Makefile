# Attention accelerator simulator.
# No external dependencies on purpose: just g++ and make. See docs/DECISIONS.md D-002.
#
#   make          build build/accel_sim
#   make test     generate the smoke golden vectors and run the tests
#   make run      run the simulator on the smoke configuration
#   make clean    remove build artifacts

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
BUILD    := build

MAIN_SRC  := src/main.cpp
LIB_SRCS  := $(filter-out $(MAIN_SRC),$(wildcard src/*.cpp))
LIB_OBJS  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRCS))

TEST_SRCS := $(wildcard tests/*.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(BUILD)/tests/%,$(TEST_SRCS))

SMOKE_VECTORS := ../vectors/smoke/q.bin

.PHONY: all test run clean vectors

all: $(BUILD)/accel_sim

$(BUILD)/accel_sim: $(BUILD)/main.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/tests/%: tests/%.cpp $(LIB_OBJS) | $(BUILD)/tests
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_OBJS)

$(BUILD) $(BUILD)/tests:
	@mkdir -p $@

# The golden vectors regenerate themselves if missing: a fresh clone must be able to run
# `make test` with no manual steps beforehand.
vectors: $(SMOKE_VECTORS)

$(SMOKE_VECTORS):
	@echo "--- generating smoke golden vectors"
	@cd .. && python3 reference/gen_vectors.py --smoke

test: $(TEST_BINS) $(SMOKE_VECTORS)
	@failed=0; \
	for t in $(TEST_BINS); do \
		echo "--- $$t"; \
		$$t || failed=1; \
		echo ""; \
	done; \
	if [ $$failed -ne 0 ]; then echo "TESTS FAILED"; exit 1; fi; \
	echo "all tests passed"

run: $(BUILD)/accel_sim $(SMOKE_VECTORS)
	./$(BUILD)/accel_sim --config ../sweeps/smoke.cfg

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
