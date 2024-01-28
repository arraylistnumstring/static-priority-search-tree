#!/bin/bash

# Execute by running
#	source constructor-comparator.sh [datatype-flag] [min-val-range] [max-val-range] \
#									 [min-num-elems] [max-num-elems] \
#									 [min-rand-seed] [max-rand-seed] \
#									 [pipe-1] [pipe-2]
#									 [constructor-tester-1] [constructor-tester-2] \
#									 [optional-output-file]
#		If optional-output-file is unspecified, output defaults to stdout

# On NYU's HPC, the folder from which sbatch is called is the working directory for the purposes of the contents of the script
mkfifo "$8" "$9"

for i in {$4..$5}
do
	for j in {$6..$7}
	do
		# Double-quotes prevent recognition of enclosed objects as separate arguments, e.g. due to presence of spaces; double-quotes are essential for proper pathname parsing; lack of double quotes around expanded elements of command-line arguments is necessary for proper recognition of separate flags to Python program
		# ${array:n:m} inputs m elements of array starting from the n-th element (1-indexed, inclusive)
		# For command-line arguments of index higher than 9, use curly braces to set them off
		"${10}" ${@:1:3} $i $j > "$8" &
		"${11}" ${@:1:3} $i $j > "$9" &
		if [ $# -lt 9 ]; then
			diff "$8" "$9"
		else
			diff "$8" "$9" > "${12}"
		fi
	done
done

rm -f "$8" "$9"
