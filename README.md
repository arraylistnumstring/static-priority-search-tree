# Static Priority Search Tree on GPU

Runs on CUDA 12.2.2; requires GCC version 7.4 or higher

Tested on NYU's Greene HPC, where accessing nvcc version 12.2.2 requires running

	singularity shell --nv /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif

and running commmands within the Singularity interface (alternatively, if only one command needs to be run within singularity, substitute `shell` with `exec` and append the desired command).


To compile testers, run:

	nvcc testers/pst-tester-driver.cu -ICommon -Ihelpers -Ipst -rdc -std=c++20 -o testers/pst-tester-driver.out

- `-I<filepath>` adds the given filepath to the list of paths to search for headers.
	- `Common`: NVIDIA CUDA sample headers
	- `helpers`: helper functions for priority search trees and related data structures
	- `pst`: PST templates
- `-rdc` allows for dynamic parallelism

For debugging options, use:
- `-DDEBUG` to toggle DEBUG preprocessor flag and corresponding print output to aid with debugging variables that are inaccessible via gdb
	- `-DDEBUG_CONSTR` to toggle constructor-internal debugging print statements
	- `-DDEBUG_TEST` to toggle pst-tester-driver.h debugging print statements
	- `-DDEBUG_WRAP` to toggle pst-test-info.h debugging print statements
- `-G` to get debugging info about device code (turns off all optimisations; is incompatible with and overrides `-lineinfo`)
- `-g` to get debugging info about host code
- `-lineinfo` to get info on which lines are causing errors

Standard is set to C++20 because Thrust has deprecated versions of C++ older than C++14 (which also allows use of simpler auto return types), constexpr if (compile-time-evaluated conditionals that may optimise away unused code) are a C++17 language feature, and requires is a C++20 language feature.


To check memory safety of GPU code, use:

	compute-sanitizer [options] [executable] [executable-options]


To use CUDA GDB, run:

	cuda-gdb [options] --args [executable] [executable-options]

- To set up a temporary directory to which cuda-gdb can output, run
	export TMPDIR=<read-write-able-directory>
in the encapsulating application

- To remedy a missing CUDA debugger driver, run
	export CUDBG_USE_LEGACY_DEBUGGER=1
which falls back to using the debugger back-end present in libcuda.so
