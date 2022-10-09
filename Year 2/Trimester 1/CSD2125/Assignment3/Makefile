PRG=prg.out
GCC=g++
GCC_FLAGS=-std=c++17 -pedantic-errors -Wall -Wextra -Werror -O2

VALGRIND_OPTIONS=-q --leak-check=full
DIFF_OPTIONS=-y --strip-trailing-cr --suppress-common-lines

SOURCE=tddaa.cpp
DRIVER=driver.cpp

MY_OUTPUT=my_output
HIS_OUTPUT=out

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(SOURCE) $(DRIVER)

0 1:
	@echo "Running test$@"
	./$(PRG) $@ > $(MY_OUTPUT)$@
	@echo "Comparing the $(MY_OUTPUT)$@ and $(HIS_OUTPUT)$@ files now"
	diff $(HIS_OUTPUT)$@ $(MY_OUTPUT)$@ $(DIFF_OPTIONS)

mem0 mem1:
	@echo "Running memory test $@"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subset mem,,$@) 1>/dev/null 2>difference$@
	@echo "Printing memory errors (if any):"; cat difference$@

clean:
	rm -f *.exe *.out *.o $(MY_OUTPUT)* $(HIS_OUTPUT)* differencemem*
	clear