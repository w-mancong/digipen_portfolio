PRG=ass6.out
GCC=g++
GCC_FLAGS=-std=c++17 -pedantic-errors -Wall -Wextra -O2 -Werror
GDB_FLAGS=-std=c++17 -pedantic-errors -Wall -Wextra -g -Werror

VALGRIND_OPTIONS=-q --leak-check=full --track-origins=yes
DIFF_OPTIONS=-y --strip-trailing-cr --suppress-common-lines
DRIVER=allocator-test.cpp

OUTPUT=my_output

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(DRIVER)

test:
	@echo "Running all tests"
	./$(PRG) > $(OUTPUT)
	diff allocator-output.txt $(OUTPUT) $(DIFF_OPTIONS)

debug:
	$(GCC) -o $(PRG) $(GDB_FLAGS) $(DRIVER)
	gdb ./$(PRG)

mem:
	@echo "Running Memory Test$@"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) 1>/dev/null 2>difference$@
	@echo "lines after this are memory errors"; cat difference$@

clean:
	rm -f *.exe *.out *.o differencemem* $(OUTPUT)*
	clear
