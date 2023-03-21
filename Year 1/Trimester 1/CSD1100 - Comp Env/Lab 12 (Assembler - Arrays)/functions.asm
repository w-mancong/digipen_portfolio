; ------------------------------------------------------------------
; File: functions.asm
; Project: CSD1100 Assignment 12
; Author: Vadim Surov, vsurov@digipen.edu
; Co-Author: Wong Man Cong, w.mancong@digipen.edu
; ------------------------------------------------------------------
    section .text

    global len
    global countn
    global counta
    global counts

;================================================================================================================================================================================
;                                                                       COUNT LENGTH OF STRING
;================================================================================================================================================================================
; count the length of the given null terminated string
len:
    mov rax, -1    ; using this to count the len of string
body1:    
    inc rax
    cmp byte [rdi + rax], 0
    jne body1
    ret    ; return rax;
;================================================================================================================================================================================
;                                                                       COUNT NUMBER
;================================================================================================================================================================================
; count the number of digits in the null terminated string
countn:
    xor r10, r10                ; using this to increment
    call len                    ; get the len of the string
    mov rcx, rax                ; moving the len of the string into rcx (counter)
    xor rax, rax                ; making rax to be 0, using this to count number of digits

num_loop:
    cmp byte [rdi + r10], 48    ; comparing curr index to see if it's '0'
    jl not_num                  ; jump to not_num if current index is not greater or equal to 48 '0'
    cmp byte [rdi + r10], 57    ; comparing curr index to see if it's '9'
    jg not_num                  ; jump to not_num if current index is not lesser or equal to 57 '9'
    jmp inc_num_len_counter
cont_num_loop:
    cmp rcx, 0
    loopnz num_loop

    ret                         ; return value stored inside rax

not_num:
    inc r10
    jmp cont_num_loop

inc_num_len_counter:
    inc r10
    inc rax
    jmp cont_num_loop
;================================================================================================================================================================================
;                                                                       COUNT ALPHABETS
;================================================================================================================================================================================
; check the total number of lower case alphabet
check_lower:
    xor r10, r10                ; using this as index
    call len                    ; get the len of the string
    mov rcx, rax                ; storing len of string into counter register
    xor rax, rax                ; using rax register as len counter

lower_loop:
    cmp byte [rdi + r10], 65    ; check current index to see if it's 'A'
    jl not_lower                ; if current index is less than 'A'
    cmp byte [rdi + r10], 90    ; check current index to see if it's 'Z'
    jg not_lower                ; if current index is more than 'Z'
    jmp inc_lower_len_counter
cont_lower_loop:
    cmp rcx, 0
    loopnz lower_loop
    
    ret                         ; return value stored inside rax
    
not_lower:
    inc r10
    jmp cont_lower_loop

inc_lower_len_counter:
    inc r10
    inc rax
    jmp cont_lower_loop

; check the total number of upper case alphabet
check_upper:
    xor r10, r10                ; using this as index
    call len                    ; get the len of the string
    mov rcx, rax                ; storing len of string into counter register
    xor rax, rax                ; using rax register as len counter

upper_loop:
    cmp byte [rdi + r10], 97    ; check current index to see if it's 'A'
    jl not_upper                ; if current index is less than 'A'
    cmp byte [rdi + r10], 122   ; check current index to see if it's 'Z'
    jg not_upper                ; if current index is more than 'Z'
    jmp inc_upper_len_counter
cont_upper_loop:
    cmp rcx, 0
    loopnz upper_loop
    
    ret                         ; return value stored inside rax
    
not_upper:
    inc r10
    jmp cont_upper_loop

inc_upper_len_counter:
    inc r10
    inc rax
    jmp cont_upper_loop

; count the total number of alphabets inside the null terminated string
counta:
    call check_lower
    mov rbx, rax
    call check_upper
    add rax, rbx
    ret                         ; return value stored inside rax
;================================================================================================================================================================================
;                                                                       COUNT SPECIAL CHARACTERS
;================================================================================================================================================================================
; counting the total number of special characters
counts:
    call counta                 ; get total count of alphabets
    mov rbx, rax                ; rbx to store count of alphabets
    call countn                 ; get total count of digits
    mov rcx, rax                ; rcx to store count of digits
    call len                    ; get the total length of string
    sub rax, rbx                ; subtract rax from rbx
    sub rax, rcx                ; subtract rax from rcx
    ret                         ; return value stored inside rax