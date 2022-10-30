/* Start Header
**************************************************************/
/*!
\file 		new-coro-lib.cpp
\author 	Lucas Nguyen, l.nguyen, 2100912
\par email: l.nguyen@digipen.edu
			2100912@sit.singaporetech.edu.sg
\date 		Oct 27 2022
\brief 		Contains an implementation of a cooperatively 
            scheduled user-level thread library under Linux.

Copyright (C) 2022 DigiPen Institute of Technology and
Singapore Institute of Technology.
Reproduction or disclosure of this file or its contents without
the prior written consent of DigiPen Institute of Technology
and Singapore Institute of Technology is prohibited.
*/
/* End Header
***************************************************************/
#include "new-coro-lib.h"
//other includes
#include <stdio.h>
#include <iostream>
#include <memory>
#include <queue>
#include <stack>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace CORO
{
    //Aliases
    using Function = void*(*)(void*);       // Function Pointer
    using Type = void*;                     // Type for Function's Return and Parameter
    using Greg = char*;                     // Register
    using Size = unsigned;                  // Size type

    // Constants
    const Size OneMB{ 1000000 };            // Size of one MB
    const ThreadID MaxThread{ 100 };            // Max number of threads
    const Size MaxStackSize{};              // Max stack size

    // Registers
    enum Registers
    {        
        REG_RAX = 0,
        REG_RCX,
        REG_RDX,
        REG_R8,
        REG_R9,
        REG_R10,
        REG_R11,
        REG_RBX,
        REG_RBP,        // Base Pointer
        REG_RDI,
        REG_RSI,
        REG_RSP,        // Stack Pointer
        REG_R12,
        REG_R13,
        REG_R14,
        REG_R15,
        REG_RIP,        // Program Counter
        REG_EFL,        // Conditional Flags
        REG_TOTAL       // Total number of registers
    };

    // Thread State
    enum ThreadState : int
    {
        NEW = 0,
        READY,
        RUNNING,
        WAITING,
        TERMINATED
    };

    // Struct for Thread Control Block
    struct TCB
    {
        ThreadID ID{};                      // Thread ID
        ThreadID ID_Wait{ MaxThread };      // Thread ID for waiting thread
        Function Task_FP{ nullptr };        // Function pointer for the task
        Type Func_Param{ nullptr };         // Function Parameter
        Type Func_Return{ nullptr };        // Function Return
        Greg Gregs[REG_TOTAL];              // Store all registers
        ThreadState State{ NEW };           // Thread's State
    };

    // Keeping track of threads
    std::atomic_uint Thread_Count{ 0 };                 // Keeps track of number of threads
    std::atomic_uint Therad_ID_Allocator{ 0 };          // Keeps track of what ID to give a thread

    // Lists and Queues
    std::unordered_map<ThreadID, TCB*> All_Threads;     // Unordered Map containing all threads
    std::queue<ThreadID> Ready_Queue;                   // Queue of threads that are ready
    std::vector<ThreadID> Waiting_List;                 // Vector of threads that are waiting
    std::vector<ThreadID> Terminated_List;              // Vector of threads that have been terminated
    std::stack<ThreadID> New_Queue;                     // Stack of new threads

    TCB *CurrThread;                  // Current Thread

    /***************************************************
    \brief
        Schedules the next thread
    \return
        void
    ***************************************************/ 
    void ScheduleThread(void);

    /***************************************************
    \brief
        Allocates stack for the given tcb
    \param tcb
        The tcb to be allocated
    \return
        void
    ***************************************************/
    void allocate_stack(TCB* tcb)
    {   
        tcb->Gregs[REG_RSP] = (Greg)malloc(OneMB) + OneMB;  // Stack Pointer
        tcb->Gregs[REG_RBP] = (Greg)malloc(sizeof(Greg));   // Frame Pointer
    }

    /***************************************************
    \brief
        Deallocates stack for the given tcb
    \param tcb
        The tcb to be deallocated
    \return
        void
    ***************************************************/
    void deallocate_stack(TCB* tcb)
    {        
        //free(tcb->Gregs[REG_RSP]);   // Stack Pointer
        //free(tcb->Gregs[REG_RBP]);   // Frame Pointer 

        // Free Pointer
        delete tcb;
    }
		
    /***************************************************
    \brief
        Initializes the primary thread.
    \return
        void
    ***************************************************/
    void thd_init(void)
    {
        // Create TCB
        TCB* primary = new TCB();

        // Set ID
        primary->ID = Thread_Count++;

        // Add to list of all threads
        All_Threads[primary->ID] = primary;

        // Set to RUNNING State
        primary->State = RUNNING;

        // Set function related variables to nullptr
        primary->Task_FP = nullptr;
        primary->Func_Param = nullptr;
        primary->Func_Return = nullptr;

        // Set Current Thread
        CurrThread = primary;
    }

    /***************************************************
    \brief
        Creates a new thread.
    \param thd_function_t
        Function pointer for the function the thread 
        will be executing.
    \param param
        Function's parameter.
    \return
        Returns the new thread's ID.
    ***************************************************/
    ThreadID new_thd(Function thd_function_t, Type param)
    {
        // Create TCB
        TCB *newThread = new TCB();

        // Set ID
        newThread->ID = Thread_Count++;

        // Set function related variables
        newThread->Task_FP = thd_function_t;
        newThread->Func_Param = param;
        newThread->Func_Return = nullptr;

        // Allocate the Stack
        allocate_stack(newThread);

        // Set to Ready
        newThread->State = NEW;

        // Add thread to "All_Threads list"
        All_Threads[newThread->ID] = newThread;
        
        // Add to newly created queue
        New_Queue.emplace(newThread->ID);

        // Return thread ID
        return newThread->ID;
    }

    /***************************************************
    \brief
        Terminates thread and yields
    \param ret_value
        Return value for the current thread
    \return
        void
    ***************************************************/
    void thread_exit(Type ret_value)
    {
        // Set Return Value
        CurrThread->Func_Return = ret_value;

        // Set status to Terminated
        CurrThread->State = TERMINATED;

        // Add to terminated queue
        Terminated_List.emplace_back(CurrThread->ID);

        // Check if have any threads waiting on this
        for(auto x = Waiting_List.begin(); x != Waiting_List.end(); ++x)
        {
            if(All_Threads.find(*x) == All_Threads.end())
                continue;

            TCB* waiter = All_Threads[*x];

            // Check if this is the waiting thread
            if(waiter->ID_Wait == CurrThread->ID)
            {   

                // Set state to ready
                waiter->State = READY;

                // Push to ready queue
                Ready_Queue.emplace(waiter->ID);

                // Erase from Waiting List
                Waiting_List.erase(x);

                break;
            }
        }

        // Yield CPU for current, schedule next
        thd_yield();
    }

    /***************************************************
    \brief
        Makes current thread wait for another thread
    \param id
        ID of thread that current thread will wait for
    \param value
        Return value of the thread that current thread
        will wait for
    \return
        Returns 1 is WAIT_SUCCESSFUL
        Returns -1 if NO_THREADS_FOUND
    ***************************************************/
   int wait_thread(ThreadID id, Type* value)
    {
        // Check if thread exists
        if(All_Threads.find(id) != All_Threads.end())
        {
            // Get next thread (Curr will wait for this)
            TCB* Waitee = All_Threads[id];
            
            // Keep waiting until Terminated
            while(Waitee->State != TERMINATED)
            {
                CurrThread->State = WAITING;
                CurrThread->ID_Wait = id;
                Waiting_List.emplace_back(CurrThread->ID);

                thd_yield();
            }

            // Assign value for return
            if(value)
                *value = Waitee->Func_Return;

            // Remove from all thread
            All_Threads.erase(id);

            // Free Waitee
            deallocate_stack(Waitee);

            // Set new current thread
            return WAIT_SUCCESSFUL;
        }
        
        // No thread with given id has been found
        return NO_THREAD_FOUND;
    }

    /***************************************************
    \brief
        Let current thread yield CPU and schedule next
        thread.
    \return
        void
    ***************************************************/
    void thd_yield(void)
    {
        // Context Saving
        // Saving registers RBX, RDI, RSI, 
            // R12, R13, R14, R15
		asm volatile
        (
            "movq %%rbx, %0 \n"
            "movq %%rdi, %1 \n"
            "movq %%rsi, %2 \n"
            "movq %%r12, %3 \n"
            "movq %%r13, %4 \n"
            "movq %%r14, %5 \n"
            "movq %%r15, %6 \n"
            : "=m" (CurrThread->Gregs[REG_RBX]),
                "=m" (CurrThread->Gregs[REG_RDI]), 
                "=m" (CurrThread->Gregs[REG_RSI]), 	  
                "=m" (CurrThread->Gregs[REG_R12]), 
                "=m" (CurrThread->Gregs[REG_R13]),
                "=m" (CurrThread->Gregs[REG_R14]),
                "=m" (CurrThread->Gregs[REG_R15])
        );
        // Saving Registers RAX, RDX, RCX, 
            // R8, R9, R10, R11
        asm volatile
        (
            "movq %%rax, %0 \n"
            "movq %%rdx, %1 \n"
            "movq %%rcx, %2 \n"
            "movq %%r8, %3 \n"
            "movq %%r9, %4 \n"
            "movq %%r10, %5 \n"
            "movq %%r11, %6 \n"	  
            : "=m" (CurrThread->Gregs[REG_RAX]),
                "=m" (CurrThread->Gregs[REG_RDX]),
                "=m" (CurrThread->Gregs[REG_RCX]),
                "=m" (CurrThread->Gregs[REG_R8]),
                "=m" (CurrThread->Gregs[REG_R9]),
                "=m" (CurrThread->Gregs[REG_R10]),
                "=m" (CurrThread->Gregs[REG_R11])
		);	
		
		// switch stack
        asm volatile
        (
            "movq %%rsp, %0 \n"
            "movq %%rbp, %1 \n"	
            : "=m" (CurrThread->Gregs[REG_RSP]),
                "=m" (CurrThread->Gregs[REG_RBP])	  
        );	
        asm volatile
        (
            "pushfq \n"
            "popq %0 \n"
            : "=m" (CurrThread->Gregs[REG_EFL])
            : 
            : "rsp"
        );	
        
        // Find next thread
        ScheduleThread();
    }

    /***************************************************
    \brief
        Schedules the next thread
    \return
        void
    ***************************************************/ 
    void ScheduleThread(void)
    {
        // Get whether ready and new queues are empty
        bool ReadyEmpty = Ready_Queue.empty();
        bool NewEmpty = New_Queue.empty();
        
        // Returns false if there are no more queues
        if(ReadyEmpty && NewEmpty)
            return;

        // Set Curr Thread to Ready if it is running
        if(CurrThread->State == RUNNING)
        {
            CurrThread->State = READY;
            Ready_Queue.push(CurrThread->ID);
        }

        // Check ready and/or new queue
        if(!ReadyEmpty)
        {   // Ready queue not empty (FCFS)
            CurrThread = All_Threads[Ready_Queue.front()];
            Ready_Queue.pop();

            // Set new current thread to running
            CurrThread->State = RUNNING;
                        
            // Context Restoring
            asm volatile
            (
                "pushq %0 \n"
                "popfq \n\t"	
                : 
                : "m" (CurrThread->Gregs[REG_EFL])
                : "cc","rsp"
            ); 

            // Switch Stack
            asm volatile
            (
                "movq %0,  %%rsp \n"
                "movq %1,  %%rbp\n"
                :	  
                : "m" (CurrThread->Gregs[REG_RSP]),
                "m" (CurrThread->Gregs[REG_RBP])	  
            );  

            // Set Context
            // Setting RBX, RDI, RSI, R12, R13, R14, R15
            asm volatile
            (
            "movq %0, %%rbx\n"		
            "movq %1, %%rdi \n"
            "movq %2, %%rsi \n"
            "movq %3, %%r12 \n"
            "movq %4, %%r13 \n"  
            "movq %5, %%r14 \n"
            "movq %6, %%r15 \n"
            : 	    
            : "m" (CurrThread->Gregs[REG_RBX]),
                "m" (CurrThread->Gregs[REG_RDI]),
                "m" (CurrThread->Gregs[REG_RSI]),
                "m" (CurrThread->Gregs[REG_R12]),
                "m" (CurrThread->Gregs[REG_R13]),
                "m" (CurrThread->Gregs[REG_R14]),
                "m" (CurrThread->Gregs[REG_R15])
            ); 

            // Setting RAX, RCX, RDX, R8, R9, R10, R11
            asm volatile
            (
            "movq %0, %%rax \n"		
            "movq %1, %%rcx \n"
            "movq %2, %%rdx \n"
            "movq %3, %%r8 \n"
            "movq %4, %%r9 \n"  
            "movq %5, %%r10 \n"
            "movq %6, %%r11 \n"
            : 	    
            : "m" (CurrThread->Gregs[REG_RAX]),
                "m" (CurrThread->Gregs[REG_RCX]),
                "m" (CurrThread->Gregs[REG_RDX]),
                "m" (CurrThread->Gregs[REG_R8]),
                "m" (CurrThread->Gregs[REG_R9]),
                "m" (CurrThread->Gregs[REG_R10]),
                "m" (CurrThread->Gregs[REG_R11])
            );
        }
        else if(!NewEmpty)
        {   // New queue not empty (LIFO)
            CurrThread = All_Threads[New_Queue.top()];
            New_Queue.pop();

            // Set new current thread to running
            CurrThread->State = RUNNING;
                    
            // Context Restoring
            asm volatile
            (
                "pushq %0 \n"
                "popfq \n\t"	
                : 
                : "m" (CurrThread->Gregs[REG_EFL])
                : "cc","rsp"
            ); 

            // Switch Stack
            asm volatile
            (
                "movq %0,  %%rsp \n"
                "movq %1,  %%rbp\n"
                :	  
                : "m" (CurrThread->Gregs[REG_RSP]),
                "m" (CurrThread->Gregs[REG_RBP])	  
            );  
            // Run thread
            thread_exit(CurrThread->Task_FP(CurrThread->Func_Param));
        }
    }
}