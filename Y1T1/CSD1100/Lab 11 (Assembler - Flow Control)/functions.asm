; ------------------------------------------------------------------
; File: functions.asm
; Project: CSD1100 Assignment 11
; Author: Vadim Surov, vsurov@digipen.edu
; Co-Author: Wong Man Cong, w.mancong@digipen.edu
; ------------------------------------------------------------------

section .text
    global f1
    global f2

; finding between two circles/cirlces and a point
f1:
    mov rax, rdi
    sub rax, rdx    ; p1-p3
    mov r10, rax

    mov rax, rsi
    sub rax, rcx    ; p2-p4
    mov r11, rax

    mov rax, r8
    add rax, r9     ; p5+p6
    mov r12, rax

    mov rax, r10    
    imul r10        ; (p1-p3)^2 (value stored in rax)
    mov r10, rax

    mov rax, r11
    imul r11        ; (p2-p4)^2 (value stored in rax)
    mov r11, rax

    mov rax, r12
    imul r12        ; (p5+p6)^2 (value stored in rax)
    mov r12, rax

    mov rax, r10
    add rax, r11    ; (p1-p3)^2 + (p2-p4)^2 (new value stored in rax)
    
    cmp rax, r12    ; comparing the two register
    jle circle_intersects

    mov rax, 0
    ret

circle_intersects:
    mov rax, 1
    ret

; summation of (p1+p3...) * (p2+p3...) + (p1+p3 - 1...) * (p2+p3 - 1...) + ...
f2:
    mov rcx, rdx
    xor r8, r8
    mov rax, rdi
    imul rsi
    add r8, rax
    cmp rcx, 0
    jz exit
    
loop_label:    
    mov rbx, rdi
    add rbx, rcx        ; (p1+p3...)
    mov rax, rbx

    mov rbx, rsi
    add rbx, rcx        ; (p2+p3...)

    imul rbx            ; (p1+p3...) * (p2+p3...)

    add r8, rax         ; (p1+p3...) * (p2+p3...) + (p1+p3 - 1...) * (p2+p3 - 1...)
    
    cmp rcx, 0
    loopnz loop_label
    
exit:
    mov rax, r8
    
    ret