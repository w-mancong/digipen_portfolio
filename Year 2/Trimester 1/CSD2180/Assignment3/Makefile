PRG=ass3_
GCC=g++
GCC_FLAGS=-std=c++14 -g

T_SOURCE=coro-lib.a
SOURCE=new-coro-lib.cpp

TEST_7_INPUT=8			#random number for testing
TEST_8_INPUT=4 4 4 10 	#random number for testing

build:
	$(GCC) -o $(PRG)t1.out $(GCC_FLAGS) ./testcases/t1.cpp $(SOURCE)
	$(GCC) -o $(PRG)t2.out $(GCC_FLAGS) ./testcases/t2.cpp $(SOURCE)
	$(GCC) -o $(PRG)t3.out $(GCC_FLAGS) ./testcases/t3.cpp $(SOURCE)
	$(GCC) -o $(PRG)t4.out $(GCC_FLAGS) ./testcases/t4.cpp $(SOURCE)
	$(GCC) -o $(PRG)t5.out $(GCC_FLAGS) ./testcases/t5.cpp $(SOURCE)
	$(GCC) -o $(PRG)t6.out $(GCC_FLAGS) ./testcases/t6.cpp $(SOURCE)
	$(GCC) -o $(PRG)t7.out $(GCC_FLAGS) ./testcases/t7.cpp $(SOURCE)
	$(GCC) -o $(PRG)t8.out $(GCC_FLAGS) ./testcases/t8.cpp $(SOURCE)

t1 t2 t3 t4 t5 t6 t7 t8:
	$(GCC) -o $(PRG)$@.out $(GCC_FLAGS) ./testcases/$@.cpp $(SOURCE)
	
expected:
	$(GCC) -o ./expected_output/e1.out $(GCC_FLAGS) ./testcases/t1.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e2.out $(GCC_FLAGS) ./testcases/t2.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e3.out $(GCC_FLAGS) ./testcases/t3.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e4.out $(GCC_FLAGS) ./testcases/t4.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e5.out $(GCC_FLAGS) ./testcases/t5.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e6.out $(GCC_FLAGS) ./testcases/t6.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e7.out $(GCC_FLAGS) ./testcases/t7.cpp $(T_SOURCE)
	$(GCC) -o ./expected_output/e8.out $(GCC_FLAGS) ./testcases/t8.cpp $(T_SOURCE)

e1 e2 e3 e4 e5 e6:
	@echo "Running expected test case $@"
	./expected_output/$@.out

e7:
	@echo "Running expected test case $@"
	./expected_output/e7.out $(TEST_7_INPUT)

e8:
	@echo "Running expected test case $@"
	./expected_output/e8.out $(TEST_8_INPUT)

1 2 3 4 5 6:
	@echo "Running test$@"
	./$(PRG)t$@.out

7:
	@echo "Running test$@"
	./$(PRG)t$@.out $(TEST_7_INPUT)

8:
	@echo "Running test$@"
	./$(PRG)t$@.out $(TEST_8_INPUT)

clean:
	rm -f *.exe *.out *.o
	clear