# Static Priority Search Tree on GPU

To compile, run:

	make

which defaults to make all. Other options include

	make <source_file_path>.o
	make <source_file_path>.d
	make <source_file_name>

where `source_file_path` is the full pathname of a source file without its file suffix, and `source_file_name` is the name of a source file without its file suffix or path prefix


For debugging options, run:

	make <target> DEBUG_FLAGS="<flags>"

where `<flags>` can be any combination of:
- `-D<preproc-flag>` to toggle various preprocessor variables that delineate debugging print statements in the code. Possible values of `<preproc-flag>`:
	- `DEBUG`: toggles print output to aid with debugging variables that are inaccessible via gdb
	- `DEBUG_CONSTR`: toggle constructor-internal debugging print statements
	- `DEBUG_TEST`: toggle test driver-internal debugging print statements
	- `DEBUG_WRAP`: toggle \*test-info.h debugging print statements
	- `CUDA_FORCE_CDP1_IF_SUPPORTED`: allow debugging of dynamic parallelism even with legacy debugger backend; not supported on devices of compute capability 9.0 or greater
		- Note: use of `cudaStreamFireAndForget` within the program is not compatible with this option
- `-G`: get debugging info about device code (turns off all optimisations; is incompatible with and overrides `-lineinfo`)
- `-g`: get debugging info about host code
- `-lineinfo`: get info about which lines are causing errors
- `-Xptxas -v`: pass the verbose option (`-v`) to the PTX optimising assembler (`-Xptxas <options>`) to get information about compiler-determined register usage and other related data


To check memory safety of GPU code, use:

	compute-sanitizer [options] [executable] [executable-options]


To use CUDA GDB, run:

	cuda-gdb [options] --args [executable] [executable-options]

- To set up a temporary directory to which cuda-gdb can output, run
	`export TMPDIR=<read-write-able-directory>`
in the encapsulating application

- To remedy a missing CUDA debugger driver, run
	`export CUDBG_USE_LEGACY_DEBUGGER=1`
which falls back to using the debugger back-end present in `libcuda.so`


On NYU's Greene HPC:

Runs on CUDA 12.2.2; requires GCC version 7.4 or higher

Accessing the CUDA 12.2.2 environment requires prepending a command with

	singularity exec --nv --overlay $SCRATCH/Isosurface-singularity-overlay-25GB-500K-files.ext3:r /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif <command>

Alternatively, to run multiple commmands within the Singularity environment, substitute `exec` with `shell`, which opens up a subshell interface for the environment.

Note:
- `--nv` uses NVIDIA drivers where applicable on the system
- `--overlay <overlay.ext3 file>:rw` uses `<overlay.ext3 file>` as the filesystem once inside of the Singularity container to create the illusion of a readable and writable filesystem within a typically read-only container
    - Note that in `rw` mode, one process has a lock on the `<overlay.ext3 file>` and no other process can use it; for shared access when running production code, open the file in read-only mode

According to HPC support staff, CUDA Singularity image version 12.3.2 may still be buggy as of 2024-07-19, so stick with version 12.2.2. Also, as of this date, compiling with 12.3.2 causes the following error:

	Error in initialising global result array index to 0 on device 0 of 1: the provided PTX was compiled with an unsupported toolchain

(A quick online search suggests this is due to NVIDIA drivers not being up-to-date).
