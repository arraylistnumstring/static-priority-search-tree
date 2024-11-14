# ?= defines variables if they have not already been defined
CXX ?= g++
# := is simple expansion (imperative-like definition)
# -ccbin specifies host compiler executable
NVCC := nvcc
NVCC_FLAGS := -ccbin $(CXX) -rdc true
CXXFLAGS := -std=c++20

# Link-time optimisation of device code; requires usage at both compile and link time
COMMON_FLAGS := -dlto
# Compile input files into object files containing relocatable device code; equivalent to -rdc true --compile
COMPILE_FLAGS := -dc
LINK_FLAGS :=

# Common: NVIDIA CUDA sample headers
# helpers: helper functions for priority search trees (PSTs) and related data structures
# pst: PST templates
INCLUDES := -ICommon -Ihelpers -Ipst
LIBRARIES :=

object_files = interval-parallel-search-tester-driver.o pst-tester-driver.o

# target: prerequisite
all: interval-parallel-search-tester-driver pst-tester-driver

# The first % matches all files with suffix .o; the second % substitutes each such match into the set of prerequisites
# $^ : all prerequisites
# $@ : file name of target of rule
%.o: $(wildcard %.c*)
	echo $(wildcard %.c*)
	$(NVCC) $(COMPILE_FLAGS) $^ -o $@

interval-parallel-search-tester-driver: interval-parallel-search-tester-driver.o

pst-tester-driver: pst-tester-driver.o

clean:
	rm -f 
