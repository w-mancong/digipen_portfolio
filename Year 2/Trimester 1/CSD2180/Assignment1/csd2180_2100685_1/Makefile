PRG=prog.out
GCC=gcc
GCC_FLAGS=-std=c11 -pedantic-errors -Wall -Wextra -Werror -O1

SOURCE=pstreea.c
VALGRIND_OPTIONS=-q --leak-check=full

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(SOURCE)

test:
	@echo "Printing process tree"
	./$(PRG)

mem0:
	@echo "running memory $@ test"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subset mem,,$@) 1>/dev/null 2>difference$@
	@echo "Printing memory erros (if any):"; cat difference$@

clean:
	rm -f *.exe *.out *.o differencemem*
	clear