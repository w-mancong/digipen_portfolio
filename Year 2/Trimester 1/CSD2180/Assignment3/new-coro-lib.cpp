#include "new-coro-lib.h"
#include <deque>
#include <algorithm>
#include <iostream>

namespace CORO
{
	enum class ThreadState
	{
		INVALID = -1,
		START,
		READY,
		RUNNING,
		WAITING,
		DONE,
	};

	enum class Register : u64
	{
		REG_RSP,
		REG_RBP,
		REG_RAX,
		REG_RBX,
		REG_RCX,
		REG_RDX,
		REG_RSI,
		REG_RDI,
		REG_R8 ,
		REG_R9 ,
		REG_R10,
		REG_R11,
		REG_R12,
		REG_R13,
		REG_R14,
		REG_R15,
		REG_RIP,
		REG_EFL,
		TOTAL_REGISTERS,
	};

	struct TCB
	{
		ThreadID id{};
		ThreadID waitID{};
		void *(*thd_function_t)(void *) { nullptr };
		void *gregs[(u64)Register::TOTAL_REGISTERS] { nullptr };
		void *args{ nullptr };	// argument values
		void *rets{ nullptr };	// return values
		ThreadState state{ ThreadState::INVALID };
	};

	namespace
	{
		/*
			declare/define the following things such as:

			stack size
			enum thread states
			struct TCB includes information such as:
				- ID
				- ID for waiting thread
				- function pointer for the task
				- registers/pointers
					-basic stack pointer
					-stack pointer
					-...
				-args/returns
				-state

			several thread lists/queues, such as
				- all threads
				- ready queue
				- waiting queue
				- partially terminated threads - need to be recycled
				- newly created threads
				- ...

			pointer to the current thread
		*/

		u64 constexpr ONE_MB{ 1'024'000 }, THREAD_SIZE{ 1'000 };
		ThreadID id_counter{ 0 };
		TCB *currTCB{ nullptr };

		std::deque<TCB *> ready{}, newly_created{};

		void RunFunction(TCB* tcb)
		{
			if(tcb->thd_function_t)
			{
				tcb->thd_function_t(tcb->args);
				thread_exit(tcb->rets);
			}
		}
	}
		
    // init the primary thread
	void thd_init(void)
	{
		/*
			 create TCB
			 add it to one list
			 
			 can set the primary thread to RUNNING directly
		*/
		currTCB = new TCB;
		currTCB->id = id_counter++;
		currTCB->state = ThreadState::RUNNING;
		newly_created.push_back(currTCB);
	}

    // creates a new thread
	ThreadID new_thd(void *(*thd_function_t)(void *), void *param)
	{
		/*
			 create TCB
			 add it to one list
			 
			 // init
			 newTCB->gregs[REG_RSP] = (char*) malloc(OneMB) + OneMB;		 
		*/
		TCB *tcb = new TCB;
		tcb->gregs[(u64)Register::REG_RBP] = new char[ONE_MB];
		tcb->gregs[(u64)Register::REG_RSP] = (char*)tcb->gregs[(u64)Register::REG_RBP] + ONE_MB;
		tcb->id = id_counter++;
		tcb->thd_function_t = thd_function_t;
		tcb->args = param;
		tcb->state = ThreadState::START;

		newly_created.push_back(tcb);

		return tcb->id;
	}

    // terminates thread and yields
	void thread_exit(void *ret_value)
	{
		/*
			add it to one list 
			if there are other threads waiting for it, put them to ready 
			yield CPU
		*/
    }

    //  wait for another thread
	s32 wait_thread(ThreadID id, void **value)
	{
	    /*
			search the thread with id
			
			yield CPU until the thread being waited for is terminated
			you can set this thread to waiting, and then yield, 
			then set to ready, when the thread with id exits
			
			obtain return value 
			
			if thread with that id is terminated, dealloc it
			
			Need to consider if the thread has already been terminated / TCB is valid
		*/
		return 0;
	}

    // let the current thread yield CPU and let the next thread run
    void thd_yield(void)
    {
		// Note that the next thread can be "ready" or newly created  
		
        // save the context!
		// context saving
		
		// save the registers
		asm volatile(
			"movq %%rbx, %0 \n"
			"movq %%rdi, %1 \n"
			"movq %%rsi, %2 \n"
			"movq %%r12, %3 \n"
			"movq %%r13, %4 \n"
			"movq %%r14, %5 \n"
			"movq %%r15, %6 \n"
			: "=m"(currTCB->gregs[(u64)Register::REG_RBX]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RDI]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RSI]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R12]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R13]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R14]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R15]));

		asm volatile(
			"movq %%rax, %0 \n"
			"movq %%rdx, %1 \n"
			"movq %%rcx, %2 \n"
			"movq %%r8, %3 \n"
			"movq %%r9, %4 \n"
			"movq %%r10, %5 \n"
			"movq %%r11, %6 \n"
			: "=m"(currTCB->gregs[(u64)Register::REG_RAX]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RDX]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RCX]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R8]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R9]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R10]),
			  "=m"(currTCB->gregs[(u64)Register::REG_R11]));

		// // switch stack
		// asm volatile
		// (
		//   "movq %%rsp, %0 \n"
		//   "movq %%rbp, %1 \n"	
		//   : "=m" (currThrd->gregs[REG_RSP]),
		// 	"=m" (currThrd->gregs[REG_RBP])	  
		// );	
		// asm volatile
		// (
		//   "pushfq \n"
		//   "popq %0 \n"
		//   : "=m" (currThrd->gregs[REG_EFL])
		//   : 
		//   : "rsp"
		// );

		asm volatile
		(
			"movq %%rsp, %0 \n"
			"movq %%rbp, %1 \n"
			: "=m"(currTCB->gregs[(u64)Register::REG_RSP]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RBP])
		);

		asm volatile
		(
			"pushfq \n"
			"popq %0 \n"
			: "=m"(currTCB->gregs[(u64)Register::REG_EFL])
			:
			: "rsp"
		);

		// // find the next thread
		// 	// load the context of the next thread!
		//     // switch stack!
		// 	asm volatile
		// 	(
		// 		"pushq %0 \n"
		// 		"popfq \n\t"
		// 		:
		// 		: "m" (currTCB->gregs[REG_EFL])
		// 		: "cc","rsp"
		// 	);

		// 	asm volatile
		// 	(
		// 		"movq %0,  %%rsp \n"
		// 		"movq %1,  %%rbp\n"
		// 		:
		// 		: "m" (currTCB->gregs[REG_RSP]),
		// 		"m" (currTCB->gregs[REG_RBP])
		// 	);

		// load the rest registers
		if(ready.size()) // ready queue is empty
		{
			ready.push_back(currTCB);
			currTCB = ready.front();
			ready.pop_front();

			asm volatile(
				"pushq %0 \n"
				"popfq \n\t"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_EFL])
				: "cc", "rsp");

			// switch stack
			asm volatile(
				"movq %0,  %%rsp \n"
				"movq %1,  %%rbp\n"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_RSP]),
				  "m"(currTCB->gregs[(u64)Register::REG_RBP]));

			asm volatile(
				"movq %0, %%rbx\n"
				"movq %1, %%rdi \n"
				"movq %2, %%rsi \n"
				"movq %3, %%r12 \n"
				"movq %4, %%r13 \n"
				"movq %5, %%r14 \n"
				"movq %6, %%r15 \n"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_RBX]),
				  "m"(currTCB->gregs[(u64)Register::REG_RDI]),
				  "m"(currTCB->gregs[(u64)Register::REG_RSI]),
				  "m"(currTCB->gregs[(u64)Register::REG_R12]),
				  "m"(currTCB->gregs[(u64)Register::REG_R13]),
				  "m"(currTCB->gregs[(u64)Register::REG_R14]),
				  "m"(currTCB->gregs[(u64)Register::REG_R15]));
			asm volatile(
				"movq %0, %%rax \n"
				"movq %1, %%rcx \n"
				"movq %2, %%rdx \n"
				"movq %3, %%r8 \n"
				"movq %4, %%r9 \n"
				"movq %5, %%r10 \n"
				"movq %6, %%r11 \n"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_RAX]),
				  "m"(currTCB->gregs[(u64)Register::REG_RCX]),
				  "m"(currTCB->gregs[(u64)Register::REG_RDX]),
				  "m"(currTCB->gregs[(u64)Register::REG_R8]),
				  "m"(currTCB->gregs[(u64)Register::REG_R9]),
				  "m"(currTCB->gregs[(u64)Register::REG_R10]),
				  "m"(currTCB->gregs[(u64)Register::REG_R11]));
			RunFunction(currTCB);
		}
		// // else next thread is new
		// 	// you can make use of the queue of newly created threads
		// 	// switch to new stack!
		// 	asm volatile
		// 	(
		// 		"pushq %0 \n"
		// 		"popfq \n\t"
		// 		:
		// 		: "m" (currTCB->gregs[REG_EFL])
		// 		: "cc","rsp"
		// 	);
		// 	asm volatile
		// 	(
		// 		"movq %0,  %%rsp \n"
		// 		"movq %1,  %%rbp\n"
		// 		:
		// 		: "m" (currTCB->gregs[REG_RSP]),
		// 		"m" (currTCB->gregs[REG_RBP])
		// 	);
		// 	// run its function, record its return val
		// 	// exit
		else
		{
			auto it = std::find_if(newly_created.begin(), newly_created.end(), [](TCB *tcb)
			{
				return tcb->id == currTCB->id; 
			});
			// add current thread control block (currTCB) to ready queue
			ready.push_back(currTCB);
			// LIFO, get the last item from the deque
			currTCB = newly_created.back();
			newly_created.pop_back();
			// remove currTCB from newly_created
			newly_created.erase(it);

			asm volatile(
				"pushq %0 \n"
				"popfq \n\t"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_EFL])
				: "cc", "rsp");

			// switch stack
			asm volatile(
				"movq %0,  %%rsp \n"
				"movq %1,  %%rbp\n"
				:
				: "m"(currTCB->gregs[(u64)Register::REG_RSP]),
				  "m"(currTCB->gregs[(u64)Register::REG_RBP]));

			RunFunction(currTCB);
		}
    }
}