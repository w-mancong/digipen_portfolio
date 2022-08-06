# ------------------------------------------------------------------
# File: functions.asm
# Project: CSD1100 Assignment 10
# Author: Vadim Surov, vsurov@digipen.edu
# Co-Author: Wong Man Cong, w.mancong@digipen.edu
# 
# @brief: basic understanding of assembly
# ------------------------------------------------------------------

    section .text

    global f1
    global f2
    global f3
    global f4
    global f5

; @brief    return the 5th parameter
f1:
    mov rax, r8
    ret

; @brief    basic addition for the first 3 parameters
f2:
    mov rax, rdi
    add rax, rsi
    add rax, rdx
    ret

; @brief    divide p3 by p4, then multiply p1 with p2. Sum them up and increment by 1. finally return the value stored in rax
f3:
    ; r15 = p3 / p4
    mov rax, rdx    ; store 3rd parameter in rax for calculations
    mov rdx, 0h     ; dividend
    mov rbx, rcx    ; store 4th parameter in rbx
    idiv rbx
    mov r15, rax    ; store it in r15 cuz got more calculations

    ; rax = p1 * p2
    xor rax, rax
    mov rax, rdi    ; store 1st parameter in rax
    mov rbx, rsi    ; store 2nd parameter in rbx
    imul rbx

    ; rax += r15
    add rax, r15 
    
    ; rax++
    inc rax

    ret

; @brief    multiply each parameter by a certain number, then add everything and return the value
f4:
    push rdx
    
    ; r11 = p1 * 100000
    mov rax, rdi            ; store 1st parameter in rax
    mov rbx, 100000
    imul rbx
    mov r11, rax

    ; r12 = p2 * 10000
    mov rax, rsi
    mov rbx, 10000
    imul rbx
    mov r12, rax

    ; r13 = p3 * 1000
    pop rax
    mov rbx, 1000
    imul rbx
    mov r13, rax

    ; r14 = p4 * 100
    mov rax, rcx
    mov rbx, 100
    imul rbx
    mov r14, rax

    ; r15 = p5 * 10
    mov rax, r8
    mov rbx, 10
    imul rbx
    mov r15, rax

    ; rax = p6 + r11 + r12 + r13 + r14 + r15
    mov rax, r9
    add rax, r11
    add rax, r12
    add rax, r13
    add rax, r14
    add rax, r15

    ret

; @brief    divide each parameter by a certain number, then subtract everything and return the value
f5:
    push rdx

    ; r11 = p1 / 100000
    mov rax, rdi    
    mov rdx, 0h     ; dividend
    mov rbx, 100000
    idiv rbx
    mov r11, rax

    ; r12 = p2 / 10000
    mov rax, rsi
    mov rdx, 0h
    mov rbx, 10000
    idiv rbx
    mov r12, rax

    ; r13 = p3 / 1000
    pop rax
    mov rdx, 0h
    mov rbx, 1000
    idiv rbx
    mov r13, rax

    ; r14 = p4 / 100
    mov rax, rcx
    mov rdx, 0h
    mov rbx, 100
    idiv rbx
    mov r14, rax

    ; r15 = p5 / 10
    mov rax, r8
    mov rdx, 0h
    mov rbx, 10
    idiv rbx
    mov r15, rax

    ; rax = r11 - r12 - r13 - r14 - r15
    mov rax, r11
    sub rax, r12
    sub rax, r13
    sub rax, r14
    sub rax, r15

    ret