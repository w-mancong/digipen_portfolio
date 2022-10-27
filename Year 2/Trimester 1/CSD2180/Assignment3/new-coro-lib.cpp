#include "new-coro-lib.h"
//other includes

namespace CORO
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
		
    // init the primary thread
	void thd_init(void)
	{
		/*
			 create TCB
			 add it to one list
			 
			 can set the primary thread to RUNNING directly
		*/
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
	int wait_thread(ThreadID id, void **value)
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
    }

    // let the current thread yield CPU and let the next thread run
    void thd_yield(void)
    {
		// Note that the next thread can be "ready" or newly created  
		
        // save the context!
		// context saving    
		
		// save the registers
		
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
	
	
   	    // // find the next thread 		
		
						
		// // if the next thread is new
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
				
		// // else 
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
						  
			//load the rest registers
    }
}