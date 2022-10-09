PRG=gcc0.exe

GCC=gcc
GCCFLAGS=-std=c11 -O2 -Wall -Wextra 

VALGRIND_OPTIONS=-q --leak-check=full
DIFFLAGS=--strip-trailing-cr -y --suppress-common-lines

OBJECTS0=readline.c
DRIVER0=driver.c

OSTYPE := $(shell uname)
ifeq ($(OSTYPE),Linux)
CYGWIN=
else
CYGWIN=-Wl,--enable-auto-import
endif

gcc0:
	$(GCC) -o $(PRG) $(CYGWIN) $(DRIVER0) $(OBJECTS0) $(GCCFLAGS)
0 1 2 3 4 5 6 7:
	@echo "running test$@"
	./$(PRG) $@ >studentout$@
	diff out$@ studentout$@ $(DIFFLAGS) > difference$@
8:
	@echo "running test$@"
	./$(PRG) $@ >studentout$@
	diff out$@ studentout$@ $(DIFFLAGS) > difference$@
9:
	@echo "running test$@"
	./$(PRG) $@ >studentout$@
	diff out$@ studentout$@ $(DIFFLAGS) > difference$@
mem0 mem1 mem2 mem3 mem4 mem5 mem6 mem7:
	@echo "running memory test $@"
	valgrind $(VALGRIND_OPTIONS) ./$(PRG) $(subst mem,,$@) 1>/dev/null 2>difference$@
clean:
	rm -f *.exe *.o student* difference*
