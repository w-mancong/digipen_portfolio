PRG = ass4.out
GCC = g++
GCC_FLAGS =-std=c++17 -pedantic-errors -Wall -Wextra -Werror -O2

VALGRIND_OPTIONS=-q --leak-check=full
DIFF_OPTIONS=-y --strip-trailing-cr --suppress-common-lines

SOURCE=wave.cpp 
DRIVER=wavetest.cpp

MY_OUTPUT=my_output
TEST_OUTPUT=test_output

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(SOURCE) $(DRIVER)

1 2:
	@echo "Running test $@"
	./$(PRG) ./wave-files/WaveTest.$@.wav
	@echo "Comparing outputs"
	diff ./wave-files/WaveTest.$@.output.wav WaveTest.wav $(DIFF_OPTIONS)

mem1 mem2:
	@echo "Running $@ test"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subset mem,,./wave-files/WaveTest.$@.wav) 1>/dev/null 2>difference$@
	@echo "Printing memory errors (if any): "; cat difference$@

clean:
	rm -f *.exe *.out *.o WaveTest.wav differencemem*
	clear