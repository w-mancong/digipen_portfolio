.text
    .globl sq_root_compute_array
    .globl sq_root_compute_varargs

.format:
    .asciz "Square root of %d is %d.\n"

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
.L1:
    movl (%rsi, %r8, 4), %eax
    movl %eax, %ebx             # %ebx is used to store array_of_elements[i] for printing later
    movl $0, %r9d               # %r9d  stores value for partial_sqrt
    movl $1, %r10d              # %r10d stores value for odd_numbers
.L2:
    subl %r10d, %eax            # res -= odd_numbers;
    incl %r9d                   # ++partial_sqrt
    addl $2, %r10d              # odd_numbers += 2;
    cmpl %r10d, %eax            # compare the line res >= odd_numbers
    jge .L2

    pushq %rdi                   # to store num_of_elements
    pushq %rsi                   # to store array_of_element
    pushq %r8
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
    jl .L1

    # before going out of stack frame, pop %rbx
    popq %rbx
    ret

sq_root_compute_varargs:
    ret
