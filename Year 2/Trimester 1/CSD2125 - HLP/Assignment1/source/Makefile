#PRG=gcc0.exe

C		 = gcc
CXX		 = g++
CFLAGS	 = -std=c11 -Wextra #-Wall
CXXFLAGS = -std=c++17 -Wextra #-Wall

VALGRIND_OPTIONS=-q --leak-check=full
DIFFLAGS=--strip-trailing-cr -y --suppress-common-lines

COBJECTS	= splitter.h splitter.c
CDRIVER		= driver.c
CXXOBJECTS 	= splitter.h splitter.cpp
CXXDRIVER	= driver.cpp

OSTYPE := $(shell uname)
ifeq ($(OSTYPE),Linux)
CYGWIN=
else
CYGWIN=-Wl,--enable-auto-import
endif

1:
	clear
	$(C) -o studentj1.out $(CYGWIN) $(CDRIVER) $(COBJECTS) $(CFLAGS)
#	./sj1.out -s 16384 -o ./split-data/test1_piece_ -i ./data/a
	./studentj1.out -s 16384 -o ./split-data/test1_piece_ -i ./data/a	

#	./sj1.out -j -o ./joined-data/test1_collected -i ./split-data/test1_piece_*
	./studentj1.out -j -o ./joined-data/test1_student_collected -i ./split-data/test1_piece_*
	diff ./joined-data/test1_student_collected ./data/a

2:
	$(CXX) -o studentj2.out $(CYGWIN) $(CXXDRIVER) $(CXXOBJECTS) $(CXXFLAGS)
	clear
#	./sj2.out -s 16384 -o ./split-data/test2_piece_ -i ./data/a
	./studentj2.out -s 16384 -o ./split-data/test2_student_piece_ -i ./data/a

#	./sj2.out -j -o ./joined-data/test2_collected -i ./split-data/test2_piece_*
	./studentj2.out -j -o ./joined-data/test2_student_collected -i ./split-data/test2_student_piece_*
	diff ./joined-data/test2_student_collected ./data/a
	
3:
	$(CXX) -c $(CXXDRIVER) $(CXXFLAGS) $(CYGWIN)
	$(C) -c $(COBJECTS) $(CFLAGS) $(CYGWIN)
	$(CXX) driver.o splitter.o -o studentj3.out

4:
	$(CXX) -o studentj4.out $(CYGWIN) $(CXXDRIVER) $(COBJECTS) $(CXXFLAGS)