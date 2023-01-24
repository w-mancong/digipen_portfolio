PRG=ass2.exe
STD_PRG=std_prg.exe
GPP=g++
GPP_FLAGS=-std=c++17 -pedantic-errors -Wall -Wextra -Werror

DRIVER=driver-bitset.cpp

VALGRIND_OPTIONS=-q --leak-check=full

STD_TEST=-DUSE_STL_BITSET

prg:
	@echo "Building my own bitset program"
	$(GPP) -o $(PRG) $(CYGWIN) $(DRIVER) $(GPP_FLAGS)

std_prg:
	@echo "Building std's bitset program"
	$(GPP) -o $(STD_PRG) $(STD_TEST) $(CYGWIN) $(DRIVER) $(GPP_FLAGS)

mine:
	./$(PRG) > my_bitset
	@echo "lines printed after diff command are mismatches with master output -- see outall"
	diff outall my_bitset --strip-trailing-cr

std:
	@echo "Using STL Bitset"
	./$(STD_PRG)

mem:
	@echo "running memory test"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subst mem,,$@) 1>/dev/null 2>difference$@
	@echo "lines after this are memory errors"; cat difference$@

clean:
	rm -f *.exe *.o *.obj diff* my_bitset *manifest*
	clear