CPP11=seq11.out
CPP17=seq17.out
GCC=g++
CPP11_FLAGS=-Wall -Werror -Wextra -Wconversion -pedantic -std=c++11 -DCPP11
CPP17_FLAGS=-Wall -Werror -Wextra -Wconversion -pedantic -std=c++17

DRIVER=main.cpp

all:
	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++11 -o seq11 main.cpp -DN=0 -DCPP11
	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++17 -o seq17 main.cpp -DN=0
	@echo "#### N=0 #### " > actual-output.txt
	./seq11 >> actual-output.txt
	./seq17 >> actual-output.txt

	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++11 -o seq11 main.cpp -DN=6 -DCPP11
	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++17 -o seq17 main.cpp -DN=6
	@echo "#### N=6 #### " >> actual-output.txt
	./seq11 >> actual-output.txt
	./seq17 >> actual-output.txt

	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++11 -o seq11 main.cpp -DN=63 -DCPP11
	g++ -Wall -Werror -Wextra -Wconversion -pedantic -std=c++17 -o seq17 main.cpp -DN=63
	@echo "#### N=63 #### " >> actual-output.txt
	./seq11 >> actual-output.txt
	./seq17 >> actual-output.txt

1 cpp11:
	$(GCC) $(CPP11_FLAGS) -o $(CPP11) $(DRIVER) -DN=0
	@echo "#### N=0 #### "
	./$(CPP11)

	$(GCC) -o $(CPP11) $(CPP11_FLAGS) $(DRIVER) -DN=6
	@echo "#### N=6 #### "
	./$(CPP11)

	$(GCC) -o $(CPP11) $(CPP11_FLAGS) $(DRIVER) -DN=63
	@echo "#### N=63 #### "
	./$(CPP11)

2 cpp17:
	$(GCC) $(CPP17_FLAGS) -o $(CPP17) $(DRIVER) -DN=0
	@echo "#### N=0 #### "
	./$(CPP17)

	# $(GCC) -o $(CPP17) $(CPP17_FLAGS) $(DRIVER) -DN=6
	# @echo "#### N=6 #### "
	# ./$(CPP17)

	# $(GCC) -o $(CPP17) $(CPP17_FLAGS) $(DRIVER) -DN=63
	# @echo "#### N=63 #### "
	# ./$(CPP17)

clean:
	rm *.out actual-output.txt *.exe 