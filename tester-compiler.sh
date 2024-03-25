#!/bin/bash

# Execute by running
#	source tester-compiler.sh [compiler-options]
# Typical options for this test suite include:
#	-G
#		Allows debuggin of device code; conflicts with and overrides -lineinfo
#	-g
#		Allows debugging of host code
#	-lineinfo
#		Provides line number information to debuggers
#	-rdc=true
#		Allows for dynamic parallelism
#	-std=c++14
#		Compiles using the C++14 standard

for file in testers/*pu*/*.c*;
do
	# Double-quoting $@ ensures that each element of the array is quoted, so that if any argument is originally double-quoted, it preserves this property (useful and potentially necessary for when arguments contain spaces)
	# Double-quoting $@ is valid no matter whether there is another character alongside it within the double quotes or not
	nvcc $file -o ${file/.c*/.out} "$@"
done
