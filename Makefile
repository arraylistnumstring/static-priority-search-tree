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
# interval-parallel-search: Liu et al.'s Marching Blocks method
# pst: PST templates
include_dirs := Common helpers interval-parallel-search pst
src := interval-parallel-search testers

# Directories in which to search for prerequisites and targets for filenames matching the pattern given
vpath %.h $(include_dirs)
vpath %.cu $(src)

INCLUDE := $(addprefix -I ,$(include_dirs))
LIBRARIES :=

sources := ips-dataset-tester-driver.cu ips-rand-data-tester-driver.cu \
	pst-dataset-tester-driver.cu pst-rand-data-tester-driver.cu


# target: prerequisite
# Sets phony targets so that make does not incorrectly assume that listed names are files, and thereby fail to run this rule if file "all" has a more recent time stamp than all of its prerequisites
.PHONY: all clean
# For each word in sources ending in .cu, replaces .cu with the empty string
all: $(sources:.cu=)

# Generate prerequisites file (makefile) <name>.du from a source file <name>.cu where each rule is of the form <stem>.o <stem>.du : <stem>.cu, followed by whatever the compiler determines to be necessary for creating that object file
# Modified from GNU make manual, section 4.14, Generating Prerequisites Automatically
%.du: %.cu
	# -e flag tells shell to exit as soon as any command fails (i.e. exits with a nonzero status)
	@set -e; \
		# Automatic variable $@ evaluates to the target
		rm -f $@; \
		# -MM flag generates prerequisites excluding system headers
		# Automatic variable $< evluates to first prerequisite
		# $$ is an escaped dollar sign, and $$ in the shell expands to the process ID, so $$$$ becomes $$ when passed to the shell, which evaluates to a process ID; in all likelihood, this is used here for generating unique filenames for a temporary file (here, <target-name>.<process-ID>)
		#	Source: https://stackoverflow.com/a/1320251
		$(NVCC) -MM $(NVCC_FLAGS) $(CXX_FLAGS) $(COMMON_FLAGS) $(COMPILE_FLAGS) $(LINK_FLAGS) $< > $@.$$$$; \
		# sed (stream editor) with command s replaces the first delimited object (a regex) with the second delimited object (an expression); the character following s is used as a delimiter, so for clarity, one uses a delimiter that would not appear in the target string, e.g. here, a comma
		# Here, as automatic variable $* is the stem of an implicit rule (which has also been captured with escaped parentheses like in vim s commands), sed replaces all trailing whitespaces and colons from <stem>.o with the target name followed by a colon; the g flag indicates that if there is more than one instance per line matching the given regex, all the strings should be thusly replaced
		# < $@.$$$$ has sed read in from the temporary file <target-name>.<process-ID>; > $@ tells sed to output to <target-name>
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

# Do not include prerequisite %.d* files if running make clean (so that such files are not generated only to be immediately destroyed by clean)
# MAKECMDGOALS contains the list of goals that was specified on the command line when calling make in the first place
# Explicitly, evaluates to true if isolating the pattern "clean" in MAKECMDGOALS returns an empty string
# Modified from section 9.2, Arguments to Specify the Goals of the GNU make manual
ifeq (,$(filter clean,$(MAKECMDGOALS)))
include $(sources:.cu=.du)
endif

clean:
	rm -f *.o $(sources:.cu=) $(sources:.cu=.du)
