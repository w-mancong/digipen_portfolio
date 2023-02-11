PRG = ass4.out
GCC = g++
GCC_FLAGS =-Wall -Wextra -Wconversion -std=c++14 -pedantic -Wno-deprecated #-Werror

VALGRIND_OPTIONS=-q --leak-check=full
DIFF_OPTIONS=-y --strip-trailing-cr --suppress-common-lines

#SOURCE=BList.cpp
DRIVER=driver-sample.cpp PRNG.cpp

MY_OUTPUT=my_output
TEST_OUTPUT=test_output

build:
	$(GCC) -o vpl_execution $(DRIVER) $(GCC_FLAGS)


clean:
	rm -f *.exe *.out *.o differencemem*
	clear