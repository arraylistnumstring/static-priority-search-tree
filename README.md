# Static Priority Search Tree on GPU

Runs on CUDA 12.2.2; requires GCC version 7.4 or higher

Tested on NYU's Greene HPC, where accessing nvcc version 12.2.2 requires running

	singularity shell --nv --overlay $SCRATCH/Isosurface-singularity-overlay-25GB-500K-files.ext3:r /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif

and running commmands within the Singularity interface (alternatively, if only one command needs to be run within singularity, substitute `shell` with `exec` and append the desired command).
Note:
	- `--nv` uses NVIDIA drivers where applicable on the system
	- `--overlay <overlay.ext3 file>:rw` uses `<overlay.ext3 file>` as the filesystem once inside of the Singularity container to create the illusion of a readable and writable filesystem within a typically read-only container
		- Note that in `rw` mode, one process has a lock on the `<overlay.ext3 file>` and no other process can use it; for shared access when running production code, open the file in read-only mode

According to HPC support staff, CUDA Singularity image version 12.3.2 may still be buggy as of 2024-07-19, so stick with version 12.2.2. Also, as of this date, compiling with 12.3.2 causes the following error:
	Error in initialising global result array index to 0 on device 0 of 1: the provided PTX was compiled with an unsupported toolchain
(a quick online search suggests this is due to NVIDIA drivers not being up-to-date).


To compile testers, run:

	nvcc testers/pst-rand-data-tester-driver.cu -ICommon -Ihelpers -Ipst -rdc true -std=c++20 -o testers/pst-rand-data-tester-driver.out

- `-I<filepath>` adds the given filepath to the list of paths to search for headers.
	- `Common`: NVIDIA CUDA sample headers
	- `helpers`: helper functions for priority search trees and related data structures
	- `pst`: PST templates
- `-rdc true` allows for dynamic parallelism

For debugging options, use:
- `-D<flag>` to toggle various preprocessor flags:
	- `-DDEBUG` to toggle DEBUG preprocessor flag and corresponding print output to aid with debugging variables that are inaccessible via gdb
	- `-DDEBUG_CONSTR` to toggle constructor-internal debugging print statements
	- `-DDEBUG_TEST` to toggle pst-rand-data-tester-driver.cu debugging print statements
	- `-DDEBUG_WRAP` to toggle pst-test-info.h debugging print statements
	- `-DCUDA_FORCE_CDP1_IF_SUPPORTED` allows for debugging of dynamic parallelism even with legacy debugger backend; note: `cudaStreamFireAndForget` inside of the program is not compatible with this option; not supported on devices of compute capability 9.0 or greater
- `-G` to get debugging info about device code (turns off all optimisations; is incompatible with and overrides `-lineinfo`)
- `-g` to get debugging info about host code
- `-lineinfo` to get info on which lines are causing errors
- `-Xptxas -v` passes the verbose option (`-v`) to the PTX optimising assembler (`-Xptxas <options>`) to get information about compiler-determined register usage and other related data

Standard is set to C++20 because Thrust has deprecated versions of C++ older than C++14 (which also allows use of simpler auto return types), constexpr if (compile-time-evaluated conditionals that may optimise away unused code) are a C++17 language feature, and requires is a C++20 language feature.

Similarly, to compile interval parallel search comparison code, run

	nvcc interval-parallel-search/ips-rand-data-tester-driver.cu -ICommon -Ihelpers -Iinterval-parallel-search -std=c++20 -o interval-parallel-search/ips-rand-data-tester-driver.out
- `interval-parallel-search` option for `-I` flag: interval parallel search testers and code
- `-D<preprocessor-variable-name>` valid options for `<preprocessor-variable-name>` (same listed effects as above for identical names):
	- `DEBUG`
	- `DEBUG_TEST`
	- `DEBUG_WRAP`
	- `DEBUG_SEARCH` to toggle search-internal debugging print statements


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
