# /*!
# file:	    square-root.s
# author:	Wong Man Cong
# email:	w.mancong\@digipen.edu
# brief:	This file contains assembly code that finds the partial sqrt root of a number
#
#		All content © 2023 DigiPen Institute of Technology Singapore. All rights reserved.
#*//*__________________________________________________________________________________*/

# Notes:
# Caller-Saved registers => %rax, %rcx, %rdx, %rsi, %rdi, %r8 - %r11 
# -> These registers have to be push/pop if I want to retrieve it's value after calling another function
# Callee-Saved registers => %rbx, %rbp, %r12 - %r15
# -> These registers need to be push/pop if I want to use them in my current function

.text
    .globl sq_root_compute_array
    .globl sq_root_compute_varargs

.format:
    .asciz "Square root of %d is %d.\n"

# C++ code to find and print the sqrt of a number
# void sq_root_compute_array(int num_of_elements, unsigned int* array_of_elements)
# {
# 	for (int i{}; i < num_of_elements; ++i)
# 	{
# 		int res = array_of_elements[i];
# 		int partial_sqrt = 0, odd_numbers = 1;
# 		while (res >= odd_numbers)
# 		{
# 			res -= odd_numbers;
# 			++partial_sqrt; odd_numbers += 2;
# 		}
# 
# 		printf("Square root of %u is %u.\n", array_of_elements[i], partial_sqrt);
# 	}
# }
sq_root_compute_array:  #int num_of_elements (%rdi), unsigned int *array_of_elements (%rsi)
    pushq %rbx
    #start of for loop
    movl $0, %r8d               # using r8 register as index i
.L0:
    movl (%rsi, %r8, 4), %eax
    movl %eax, %ebx             # %ebx is used to store array_of_elements[i] for printing later
    movl $0, %r9d               # %r9d  stores value for partial_sqrt
    movl $1, %r10d              # %r10d stores value for odd_numbers
.L1: # while loop portion
    subl %r10d, %eax            # res -= odd_numbers;
    incl %r9d                   # ++partial_sqrt
    addl $2, %r10d              # odd_numbers += 2;
    cmpl %r10d, %eax            # compare the line res >= odd_numbers
    jge .L1

    # pushing these registers because it might be overwritten
    pushq %rdi                  # to store num_of_elements
    pushq %rsi                  # to store array_of_element
    pushq %r8                   # to store index i 
    # printf here
    movq $.format, %rdi
    movq %rbx, %rsi
    movq %r9, %rdx
    xorq %rax, %rax
    call printf
    popq %r8
    popq %rsi
    popq %rdi

    incl %r8d
    # i < num_of_elements 
    cmpl %edi, %r8d
    jl .L0

    # before going out of stack frame, pop %rbx
    popq %rbx
    ret

sq_root_compute_varargs:
    # Prologue
    pushq %rbp
    movq %rsp, %rbp

    pushq %r9
    pushq %r8
    pushq %rcx
    pushq %rdx
    pushq %rsi
    pushq %rdi                      # Pushing these registers so that I can manipulate it with %rsp

    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %rbx                      # Pushing these registers to use for calculating the sqrt

    # Code in between here are used to calculate sqrt
    addq $40, %rsp                  # Offset %rsp by 40 bytes to let it point to %rdi
    xorq %r12, %r12                 # %r12 will be used as my index
.LS0:
    movq (%rsp, %r12, 8), %r13      # %r13 represents res
    cmpq $0, %r13                   # jumps to .LS4 if %r13 is 0
    jz .LS4

    movq %r13, %rbx                 # use for printf later
    # Calculate sqrt of this number
    xorq %r14, %r14                 # %r14 represents partial_sqrt = 0
    movq $1, %r15                   # %r15 represents odd_numbers
.LS1:# while loop
    subq %r15, %r13                 # res -= odd_numbers
    incq %r14                       # ++partial_sqrt
    addq $2, %r15                   # odd_numbers += 2
    cmpq %r15, %r13                 # res >= odd_numbers
    jge .LS1

    # printf here
    movq $.format, %rdi
    movq %rbx, %rsi
    movq %r14, %rdx
    xorq %rax, %rax
    call printf

    # Increment counter and to offet %rsp accordingly so that %13 can store argument values
    incq %r12
    cmpq $6, %r12
    jne .LS0 

    # Reaches the end of the 6 registers, need to offset %rsp to make it point to higher address
    addq $64, %rsp                  # Offset %rsp by 64 bytes so that it now points to callee function stack memory
    xorq %r12, %r12                 # reset %r12 to be 0
.LS2:
    movq (%rsp, %r12, 8), %r13
    cmpq $0, %r13                   # jumps to .LS4 if %r13 is 0
    jz .LS4

    movq %r13, %rbx
    # Calculate sqrt of this number
    xorq %r14, %r14                 # %r14 represents partial_sqrt = 0
    movq $1, %r15                   # %r15 represents odd_numbers
.LS3:# while loop
    subq %r15, %r13                 # res -= odd_numbers
    incq %r14                       # ++partial_sqrt
    addq $2, %r15                   # odd_numbers += 2
    cmpq %r15, %r13                 # res >= odd_numbers
    jge .LS3

    # printf here
    movq $.format, %rdi
    movq %rbx, %rsi
    movq %r14, %rdx
    xorq %rax, %rax
    call printf

    # Increment counter and to offet %rsp accordingly so that %13 can store argument values
    incq %r12
    cmpq $2, %r12
    jle .LS2

.LS4:    
    popq %rbx                       # Restoring the values for these registers before ending the code
    popq %r15
    popq %r14
    popq %r13
    popq %r12

    popq %rdi
    popq %rsi
    popq %rdx
    popq %rcx
    popq %r8
    popq %r9

    # Epilogue
    movq %rbp, %rsp
    popq %rbp
    ret
