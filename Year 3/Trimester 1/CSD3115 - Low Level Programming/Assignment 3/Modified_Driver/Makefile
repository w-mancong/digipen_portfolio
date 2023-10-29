CC = g++
CFLAGS = -O2
COLOUR_RED=\033[1;31m
COLOUR_GREEN=\033[0;32m
COLOUR_YELLOW=\033[1;33m
COLOUR_BLUE=\033[1;36m
END_COLOUR=\033[0m

all: opt_poly.c poly.c clock.h compute_least_square.c

	@echo "$(COLOUR_YELLOW)Starting makefile$(END_COLOUR)"
	@echo ---------------------------------------------------------------------------------------------------
	
	# @echo "$(COLOUR_BLUE)Ensuring g++-multilib is installed$(END_COLOUR)"
	# sudo apt install g++-multilib
	# @echo "$(COLOUR_GREEN)Installing complete!$(END_COLOUR)"
	# @echo ---------------------------------------------------------------------------------------------------
	
	@echo "$(COLOUR_BLUE)Compiling x64/64-bit assembly... (for data flow/comparison.pdf)$(END_COLOUR)"
	g++ -m64 -O2 opt_poly.c -S -fverbose-asm -o 64bit_asm.s
	@echo "$(COLOUR_GREEN)x64 Assembly Code Compiled!$(END_COLOUR)"
	
	@echo "$(COLOUR_BLUE)Running x64/64-bit version (for graph.pdf)$(END_COLOUR)"
	g++ -m64 -O2 -DPOLY_OPT poly.c opt_poly.c compute_least_square.c -lm -o 64bit_vpl_execution
	./64bit_vpl_execution
	@echo "$(COLOUR_GREEN)64bit_vpl_execution executed successfully!$(END_COLOUR)"
	
	@echo ---------------------------------------------------------------------------------------------------
	@echo "$(COLOUR_BLUE)Compiling x86/32-bit assembly... (for data flow/comparison.pdf)$(END_COLOUR)"
	g++ -m32 -O2 opt_poly.c -S -fverbose-asm -o 32bit_asm.s
	@echo "$(COLOUR_GREEN)x86 Assembly Code Compiled!$(END_COLOUR)"
	
	@echo "$(COLOUR_BLUE)Running x86/32-bit version (for graph.pdf)$(END_COLOUR)"
	g++ -m32 -O2 -DPOLY_OPT poly.c opt_poly.c compute_least_square.c -lm -o 32bit_vpl_execution
	./32bit_vpl_execution
	@echo "$(COLOUR_GREEN)32bit_vpl_execution executed successfully!$(END_COLOUR)"
	
	@echo ---------------------------------------------------------------------------------------------------
	
	@echo "$(COLOUR_RED)[ERROR!] Just Kidding, if you reach here, all supporting documents here should help!$(END_COLOUR)"
	@echo "$(COLOUR_YELLOW)All the best to compiling on Moodle! :D$(END_COLOUR)"
	
clean:
	rm -f  *.o *~