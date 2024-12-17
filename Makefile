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

# Recursive assignment (essentially like call by name parameters, where the text assigned to the variable is substituted in its entirety each time it is called and evaluated only when used) so that all capital-letter variables can be grouped together, even though the following capital-letter variables depend on variables with lowercase names that are defined later
INCLUDE = $(addprefix -I ,$(include_dirs))
LIBRARIES = $(addprefix -l ,$(libraries))

# File suffixes of source files
suffixes := .c .cpp .cu

# Set all directories that do not begin with a period to be an include directory
# As maxdepth is a global option, the shell issues a complaint if it is not placed before non-global options
include_dirs := $(shell find . -maxdepth 1 ! -name '.*' -type d)
libraries := GL GLU
# Find all source files by finding all filenames ending in one of the suffixes found in variable suffixes
source_files := $(foreach suffix_type,$(suffixes),$(shell find . -name '*$(suffix_type)' -type f))

# Find all driver files (i.e. files that need to be compiled into executables) by picking out all source_files that have a main() function
# Use {} so that make is not confused by the open parenthesis in the regex given to grep and thus would not mistakenly think that the shell call has not been closed properly
driver_files := ${shell grep 'main[[:space:]]*(' $(source_files) -l}
# Add .o suffix to all source file names to form object file names to avoid cross-suffix overlap (e.g. <name>.c and <name>.cpp both becoming <name>.o)
object_files := $(addsuffix .o,$(source_files))
# Remove source file suffixes from names in driver_file to get executable names
# For each type of source file suffix, find the names in driver_files with that suffix and generate the corresponding executable name; then, notdir strips all files of their source directory names, placing executables in the top-level project folder
executables  := $(notdir \
					$(foreach suffix_type,$(suffixes), \
						$(patsubst %$(suffix_type),%,$(filter %$(suffix_type),$(driver_files))) \
					) \
				)

# As vpath is a directive, use this method of variable definition to parameterise the calling of vpath
# Use of define allows for newlines to be included in variable values (see section 6.8 of the GNU Make manual, Defining Multi-Line Variables)
# $(n) access the n-th parameter (1-indexed)
define vpath_func
	vpath $(1) $(2)
endef

# Add include_dirs to the search path of all source files (files with suffix found in variable suffixes) as well as all header files
# Use of eval allows for creation of non-constant Makefile syntax, including but not limited to targets, implicit or explicit rules, new make variables, etc.
# Use of call is necessary for vpath_func to be evaluated as a function with the parameters listed, rather than as a variable
$(foreach suffix_tpe,$(suffixes) .h,$(eval $(call vpath_func,%$(suffix_type),$(include_dirs))))






# All prerequisites for a single target are put together, so adding new targets to .PHONY does not overwrite previous values
.PHONY: clean
clean:
	rm -f $(object_files) $(executables)




# All commands preceding this line have been tested for correctness









blank :=
# Use $(blank) to enforce the presence of a newline, e.g. so that automake will not remove the second newline silently
define newline

$(blank)
endef

# target: prerequisite
# Sets phony targets so that make does not incorrectly assume that listed names are files, and thereby fail to run this rule if file "all" has a more recent time stamp than all of its prerequisites
.PHONY: all clean
# For each word in sources ending in .cu, replaces .cu with the empty string
all: $(foreach suffix,$(suffixes), \
		$(subst $(suffix),,$(filter %.$(suffix),$(sources))))

# Canned recipe; for use in generating prerequisites file (makefile) <name>.d<remainder-of-suffix> from a source file <name>.c<remainder-of-suffix> where each rule is of the form <stem>.o <stem>.d<remainder-of-suffix>: <stem>.c<remainder-of-suffix>, followed by whatever the compiler determines to be necessary for creating that object file
# Modified from GNU make manual, section 4.14, Generating Prerequisites Automatically
define create_prereq
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
endef

# Make a prerequisites file with a choice of filename suffix based on the suffix of the original source file
$(foreach suffix,$(suffixes), \
	$(subst .c,.d,%.$(suffix)): %.$(suffix); $(create_prereq) $(newline))

# Do not include prerequisite %.d* files if running make clean (so that such files are not generated only to be immediately destroyed by clean)
# MAKECMDGOALS contains the list of goals that was specified on the command line when calling make in the first place
# Explicitly, evaluates to true if isolating the pattern "clean" in MAKECMDGOALS returns an empty string
# Modified from section 9.2, Arguments to Specify the Goals of the GNU make manual
ifeq (,$(filter clean,$(MAKECMDGOALS)))
# In each iteration over the list .c .cpp .cu, suffix takes on the value of the list that is active in the current iteration, then finds the words in sources matching the pattern <stem>.<suffix> and substitutes that suffix with .d
include $(foreach suffix,$(suffixes), \
	$($(subst .c,.d,$(filter %.$(suffix),sources))))
endif
