PRG=quiz3.out
GCC=gcc
GCC_FLAGS=-pthread

SOURCE=linearSearch.c

build:
	$(GCC) -o $(PRG) $(GCC_FLAGS) $(SOURCE)

test:
	@echo "Running linearSearch"
	./$(PRG) < test.txt

clean:
	rm -f *.exe *.out *.o
	clear