CMAKE ?= cmake
BUILD ?= build
CMAKE_FLAGS ?=
JOBS ?= 4

.PHONY: all test clean
all:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)
	$(CMAKE) --build $(BUILD) --parallel $(JOBS)

test: all
	$(CMAKE) --build $(BUILD) --target test

clean:
	$(CMAKE) --build $(BUILD) --target clean
