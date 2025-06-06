# Minimal maintenance Makefile that automatically searches for source files, compiles them into object files, and links those into executables, all with auto-determined dependencies; only the choice of compiler, flags and file-type suffixes need to be updated for new projects
# Object files are constructed from each source file; dependencies are determined with the help of the compiler's -MM option
# Executables are constructed from any object files whose source files contain a main function; an executable file's dependencies are the object file versions of any source files found in the set of dependencies for the executable's input object file

# Author: Brian H. Chiang



# ?= defines variables if they have not already been defined
CXX ?= g++
# Standard is set to C++20 because:
#	Thrust has deprecated versions of C++ older than C++14
#	C++14 allows use of simpler auto return types
#	C++17 introduced constexpr if (compile-time-evaluated conditionals that may optimise away unused code)
#	C++20 introduced requires (allows specification of additional restrictions on template parameters)
CXXFLAGS := -std=c++20
# := is simple expansion (imperative-like definition)
# -ccbin specifies host compiler executable
NVCC := nvcc
NVCC_FLAGS := -ccbin $(CXX) $(CXXFLAGS)

# COMMON_FLAGS for flags used at both compile-time and link-time
# -dlto: link-time optimisation of device code; requires usage at both compile and link time
COMMON_FLAGS := -dlto
# For debugging flags, including those specified in preprocessor commands in the body of the code; to be overridden from the command line, as some of the debugging flags disable optimisations
DEBUG_FLAGS :=
# -dc: compile input files into object files containing relocatable device code; equivalent to -rdc true --compile
# -rdc true allows for dynamic parallelism, as well as optimisable linkage across multiple translation units
# Recursive assignment necessary for contained recursively assigned variables to be properly expanded, no matter the use location of such contained variables relative to their definitions
COMPILE_FLAGS = $(COMMON_FLAGS) $(DEBUG_FLAGS) $(INCLUDE) -dc
LINK_FLAGS = $(COMMON_FLAGS) $(LIBRARIES)

# Recursive assignment (essentially like call by name parameters, where the text assigned to the variable is substituted in its entirety each time it is called and evaluated only when used) so that all capital-letter variables can be grouped together, even though the following capital-letter variables depend on variables with lowercase names that are defined later
INCLUDE = $(addprefix -I ,$(include_dirs))
LIBRARIES = $(addprefix -l ,$(libraries))

# File suffixes of header and source files
header_suffixes := .h
depend_suffix := .d
object_suffix := .o
source_suffixes := .c .cpp .cu

# Set all directories that do not begin with a period to be an include directory (including symbolically linked directories)
# As maxdepth is a global option, the shell issues a complaint if it is not placed before non-global options
# Remove ./ prefix from directory names
include_dirs := $(patsubst ./%,%,$(shell find . -maxdepth 1 ! -name '.*' \( -type d -o -type l \)))
libraries := GL GLU
# Find all source files by finding all filenames ending in one of the suffixes found in source_suffixes that is not contained in a hidden folder
# Use of $\ ensures that no whitespace is added, as the escaped newline is evaluated first, becoming a space, then "$ " evaluates to the empty string, as the variable " " has no value (similar reasoning applies to not having a space before $\)
source_files := $(patsubst ./%,%,$\
					$(foreach source_suffix,$(source_suffixes),$\
						$(shell find . -name '*$(source_suffix)' ! -path './.*' -type f)$\
					)$\
				)

# Save prerequisites in generated makefiles so that they need not be computed every time, but only when the makefile does not already exist or the corresponding source file is updated
depend_files := $(foreach source_suffix,$(source_suffixes),$\
					$(patsubst %$(source_suffix),%$(depend_suffix),$\
						$(filter %$(source_suffix),$(source_files))$\
					)$\
				)
# Find all driver files (i.e. files that need to be compiled into executables) by picking out all source_files that have a main() function
# Use {} so that make is not confused by the open parenthesis in the regex given to grep and thus would not mistakenly think that the shell call has not been closed properly
driver_files := ${shell grep 'main[[:space:]]*(' $(source_files) -l}
# Replace source file suffixes with object_suffix to match requirements auto-generated by -MM flag of compiler
object_files := $(foreach source_suffix,$(source_suffixes),$\
					$(patsubst %$(source_suffix),%$(object_suffix),$\
						$(filter %$(source_suffix),$(source_files))$\
					)$\
				)
# Remove source file suffixes from names in driver_file to get executable names
# For each type of source file suffix, find the names in driver_files with that suffix and generate the corresponding executable name; then, notdir strips all files of their source directory names, placing executables in the top-level project folder
executables := $(notdir \
					$(foreach source_suffix,$(source_suffixes),$\
						$(patsubst %$(source_suffix),%,$(filter %$(source_suffix),$(driver_files)))$\
					)$\
				)

# As vpath is a directive, use this method of variable definition to parameterise the calling of vpath
# Use of define allows for newlines to be included in variable values (see section 6.8 of the GNU Make manual, Defining Multi-Line Variables)
# $(n) access the n-th parameter (1-indexed)
define vpath_func
	vpath $(1) $(2)
endef

# Add include_dirs to the search path of all source files (files with suffix found in source_suffixes) as well as all header files
# Use of eval allows for creation of non-constant Makefile syntax, including but not limited to targets, implicit or explicit rules, new make variables, etc.
# Use of call is necessary for vpath_func to be evaluated as a function with the parameters listed, rather than as a variable
$(foreach suffix_type,$(depend_suffix) $(header_suffixes) $(source_suffixes),$\
	$(eval $(call vpath_func,%$(suffix_type),$(include_dirs))))

.PHONY: all
all: $(executables)

# Prerequisites for dependency files; find the source file whose filename (minus the suffix) matches that of depend_file exactly
# All prerequisites for a single target are put together, so specifying prerequisites for a target multiple times does not overwrite previous values
$(foreach depend_file,$(depend_files),$\
	$(eval $(depend_file): \
		$(foreach source_suffix,$(source_suffixes),$\
			$(filter $(depend_file:$(depend_suffix)=$(source_suffix)),$(source_files))$\
		)$\
	)$\
)
# To be exact with dependency file generation (as only filenames found in $(depend_files) are used in include directives or cleaned up by make clean), specify $(depend_files) as target instead of %$(depend_suffix) (though in this particular case, both work)
# Double quotes allow for interpolation of content contained within (whereas single quotes preserve the literal value of everything they contain, including $,\, etc.); use either to preserve the presence of literal backslashes
# As make uses whitespace as a token separator, generate the desired regex (which requires a replacement of whitespace with the or operator (|)) by using shell functions
$(depend_files):
	$(call gen-prereqs,$(shell echo -n "($$(echo -n $(addprefix \\,$(source_suffixes)) | tr ' ' '|'))\\>"))

define gen-prereqs
@# @ prefix in a recipe prevents echoing of that line (i.e. outputting the contents of a command before executing it)
@# Generate and place object file prerequisites in the associated source file's dependency file
@# Generate list of prerequisites with a direct command, rather than saving its result as a function parameter; this is because as a function parameter, it may exceed MAX_ARG_STRLEN (the maximal length of a single argument to the shell, e.g. when passed to echo) and fail to properly write to the dependencies file
@# -MM: generate prerequisites for object file created from input source file; overridden by actual compilation into an object file if -o option is specified
@# As -MM option automatically places object file target in current directory, prepend the directory of the source file to match the names specified in object_files
echo -n $(dir $<) > $@
$(NVCC) $(NVCC_FLAGS) $(COMPILE_FLAGS) -MM $< >> $@
echo >> $@

@# If source file prerequisite is a driver source file, add the object file prerequisites for the associated executable; additionally, as each executable depends on different object files, write rules and recipes for target executables in the corresponding dependency file
@# -n checks for the non-nullity of a string
@# By default, make evaluates each line of a recipe in a different shell (and the .ONESHELL variable forces all recipes to use a single shell per recipe throughout the makefile), so use shell newlines to force use of the same shell and thereby keep values of instantiated shell variables, while not causing unforseen consequences by modifying the globally effective variable .ONESHELL
@# Build executable in top-level directory
if [ -n "$(filter $<,$(driver_files))" ]; \
then \
	echo >> $@; \
	executable=$(notdir $(@:$(depend_suffix)=)); \
	echo "$$executable: \\" >> $@; \
	prereq_objs=$$($(NVCC) $(NVCC_FLAGS) $(COMPILE_FLAGS) -MM $< | \
		grep -E "$(1)" | \
		sed -E 's/.*([[:space:]][^[:space:]]*)$(1) \\/\1$(object_suffix)/'); \
	echo "\t$$prereq_objs" >> $@; \
	echo "\t\$$(NVCC) \$$(NVCC_FLAGS) \$$(LINK_FLAGS) $$prereq_objs -o $$executable" >> $@; \
fi
endef


# For each object file, use as input file the prerequisite that has the same filename prefix as the target and has a source file suffix (this is because while object files are the result of compiling single source files, if any header has its own source file(s) that are thus part of this object file's prerequisites, that header's source file(s) should not be sent as input when compiling this object file)
# As vpath is only meant for searching for existing header and source files, generated files that are targets must be specified if they are found in some directory other than the top-level one
$(object_files):
	$(NVCC) $(NVCC_FLAGS) $(COMPILE_FLAGS) -o $@ \
		$(filter $(wildcard $(@:$(object_suffix)=)*),$\
			$(filter $(foreach source_suffix,$(source_suffixes),%$(source_suffix)),$^))


.PHONY: clean
clean:
	# -command causes make to ignore errors that arise when executing command
	-rm -f $(depend_files)
	-rm -f $(executables)
	-rm -f $(object_files)


# Only include dependency files if not running make clean (as such files would be generated because of the include directive just to be immediately removed afterwards by the recipe in target clean)
# Note that all commands in an included Makefile are run as if they were directly written in the current Makefile
# MAKECMDGOALS contains the goal specified on the command line when invoking make, e.g. make all has goal "all"
ifeq (,$(filter clean,$(MAKECMDGOALS)))
include $(depend_files)
endif
