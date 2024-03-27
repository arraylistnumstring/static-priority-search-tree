#!/bin/bash

# Execute by running
#	source constructor-comparator.sh [tester-executable] \
#									 [pipe-1] [pipe-2] \
#									 [pst-type-1] [pst-type-2] \
#									 [-o optional-output-file] \
#									 [remaining command-line arguments to tester] \
#									 [optional-output-file]
#		If optional-output-file is unspecified, output defaults to stdout

# On NYU's HPC, the folder from which sbatch is called is the working directory for the purposes of the contents of the script
mkfifo "$4" "$5"

for i in {$4..$5}
do
	for j in {$6..$7}
	do
		# Double-quotes prevent recognition of enclosed objects as separate arguments, e.g. due to presence of spaces; double-quotes are essential for proper pathname parsing; lack of double quotes around expanded elements of command-line arguments is necessary for proper recognition of separate flags to Python program
		# ${array:n:m} inputs m elements of array starting from the n-th element (1-indexed, inclusive)
		# For command-line arguments of index higher than 9, use curly braces to set them off
		# Output file given
		if [ "$6" = "-o"]; then
			"${1}" ${@:1:3} $i $j > "$1" &
			"${1}" ${@:1:3} $i $j > "$2" &
			diff "$1" "$2" > "$7"
		else
			"${1}" ${@:1:3} $i $j > "$1" &
			"${1}" ${@:1:3} $i $j > "$2" &
			diff "$1" "$2"
		fi
	done
done

rm -f "$1" "$2"
