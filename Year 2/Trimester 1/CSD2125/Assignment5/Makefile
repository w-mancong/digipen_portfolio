PRG=ass5.out
GCC=g++
GCC_FLAGS=-std=c++17 -pedantic-errors -Wall -Wextra -O2 -Werror

VALGRIND_OPTIONS=-q --leak-check=full --track-origins=yes
DIFF_OPTIONS=-y --strip-trailing-cr --suppress-common-lines

DRIVER=matrix-proxy-driver.cpp

OUTPUT=my_output

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(DRIVER)

0 1 2 3 -all:
	@echo "Running Test$@"
	./$(PRG) $@ > $(OUTPUT)$@
	@echo "Comparing outputs"
	diff out$@ $(OUTPUT)$@ $(DIFF_OPTIONS)

mem0 mem1 mem2 mem3 mem-all:
	@echo "Running Memory Test$@"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subst mem,,$@) 1>/dev/null 2>difference$@
	@echo "lines after this are memory errors"; cat difference$@

clean:
	rm -f *.exe *.out *.o differencemem* $(OUTPUT)*
	clear