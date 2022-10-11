PRG=ass2.out
DEBUG=ass2_debug.out
GCC=gcc
GCC_FLAGS=-std=c11 -pedantic-errors -Wall -Wextra -O2 -Werror
DEBUG_FLAGS=-std=c11  -pedantic-errors -Wall -Wextra -Werror 

SOURCE=uShell.c
VALGRIND_OPTIONS=-q --leak-check=full

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(SOURCE) -lm
	ls

db: #debug build
	$(GCC) -g -o $(DEBUG) $(DEBUG_FLAGS) $(SOURCE) -lm
	ls

debug:
	@echo "Debugging..."
	gdb ./$(DEBUG)

core:
	@echo "Debugging w/ core..."
	gdb ./$(DEBUG) ./core

test:
	@echo "Running uShell program"
	./$(PRG)

mem:
	@echo "running memory test"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subst mem,,$@) 1>/dev/null 2>difference$@
	@echo "Printing memory errors (if any): "; cat difference$@

clean:
	rm -f *.exe *.out *.o differencemem*
	clear