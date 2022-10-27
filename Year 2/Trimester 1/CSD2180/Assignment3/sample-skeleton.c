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
			- ...
			

		pointer to the current thread
	*/
		
    // init the primary thread
    init
    {
		/*
			 create TCB
			 add it to one list
		*/
    }

    // creates a new thread
    new
    {
		/*
			 create TCB
			 add it to one list
		*/
    }

    // terminates thread and yields
    exit
    {
		/*
			add it to one list 
			if there are other threads waiting for it, put them to ready 
			yield CPU
		*/
    }

    //  wait for another thread
   wait
    {
	    /*
			search the thread with id
			
			yield CPU until the thread being waited for is terminated  
			
			obtain return value 
			
			if thread with that id is terminated, dealloc it
			
			Need to consider if the thread has already been terminated / TCB is valid
		*/
    }

    // let the current thread yield CPU and let the next thread run
    yield
    {
        // save the context!
   	    // find the next thread 		
		
						
		// if the next thread is new
			// switch to new stack!
			// run its function, record its return val
			// exit
				
		// else 
			// load the context of the next thread!
            // switch stack!         
    }
}