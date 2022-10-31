/*!*****************************************************************************
\file new-coro-lib.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Operating System
\par Assignment 3
\date 31-10-2022
\brief
An implementation of a user level thread library that does concurrency
*******************************************************************************/
#include "new-coro-lib.h"
#include <deque>
#include <unordered_map>
#include <algorithm>

#define REG_RAX	 	0
#define REG_RBX	 	1
#define REG_RCX	 	2
#define REG_RDX	 	3
#define REG_RDI	 	4
#define REG_RSI	 	5
#define REG_R8	 	6
#define REG_R9	 	7
#define REG_R10	 	8
#define REG_R11	 	9
#define REG_R12	 	10
#define REG_R13	 	11
#define REG_R14	 	12
#define REG_R15	 	13
#define REG_RBP 	14	 
#define REG_RSP 	15
#define REG_RIP 	16
#define REG_EFL 	17

namespace CORO
{
	enum ThreadState : int
	{
		INVALID = -1,
		NEW,
		READY,
		RUNNING,
		WAITING,
		TERMINATED,
	};

	struct TCB
	{
		ThreadID id{};
		ThreadID waitID{};
		void *(*thd_function_t)(void *){nullptr};
		void *param{nullptr}; 		// argument values
		void *ret_value{nullptr}; 	// return values
		ThreadState state = ThreadState::INVALID;
		void *gregs[18]{nullptr};
	};

	namespace
	{
		size_t constexpr ONE_MB{ 1'024'000 };
		static ThreadID id_counter{ 0 };
		TCB *currTCB{ nullptr };

		std::deque<ThreadID> ready_queue{}, new_stack{}, waiting_list{};
		std::unordered_map<ThreadID, TCB *> all_threads{};

		/*!*****************************************************************************
			\brief A wrapper function to help the user call the thread_exit function
		*******************************************************************************/
		void RunFunction(void)
		{
			if(currTCB->thd_function_t)
				thread_exit(currTCB->thd_function_t(currTCB->param));
		}
	}

	/*!*****************************************************************************
		\brief Initialise user-level thread library
	*******************************************************************************/
	void thd_init(void)
	{
		TCB *tcb = new TCB;
		tcb->id = id_counter++;
		tcb->state = ThreadState::RUNNING;

		all_threads[tcb->id] = tcb;
		currTCB = tcb;
	}

	/*!*****************************************************************************
		\brief Create a new thread

		\param [in] thd_function_t: Function to be called by thread library
		\param [in] param: param for thd_function_t
	*******************************************************************************/
	ThreadID new_thd(void *(*thd_function_t)(void *), void *param)
	{
		TCB *tcb = new TCB;
		tcb->id = id_counter++;
		tcb->thd_function_t = thd_function_t;
		tcb->param = param;
		tcb->ret_value = nullptr;
		tcb->state = ThreadState::NEW;

		tcb->gregs[REG_RBP] = new char[ONE_MB];
		tcb->gregs[REG_RSP] = (char *)tcb->gregs[REG_RBP] + ONE_MB;

		new_stack.push_back(tcb->id);
		all_threads[tcb->id] = tcb;
		return tcb->id;
	}

	/*!*****************************************************************************
		\brief Terminate the thread and store the return value of the thd_function_t
			   into ret_value

		\param [in] ret_value: return value provided by thd_function_t
	*******************************************************************************/
	void thread_exit(void *ret_value)
	{
		currTCB->ret_value = ret_value;
		currTCB->state = TERMINATED;

		// Search to see if any threads are waiting for currTCB
		for(ThreadID const& id : waiting_list)
		{
			TCB *tcb = all_threads[id];
			if(tcb->waitID != currTCB->id)
				continue;
			tcb->state = READY;
			ready_queue.push_back(tcb->id);

			auto it = std::find(waiting_list.begin(), waiting_list.end(), tcb->id);
			waiting_list.erase(it);

			break;
		}

		thd_yield();
	}

	/*!*****************************************************************************
		\brief Waits for a thread to terminate and get the return value

		\param [in] id: id of thread for the current thread to wait for
		\param [in] value: stores the return value after thread id has terminated

		\return NO_THREAD_FOUND if no threads with id are found, else WAIT_SUCCESSFUL
	*******************************************************************************/
	int wait_thread(ThreadID id, void **value)
	{
        auto it = all_threads.find(id);
        if(it == all_threads.end())
            return NO_THREAD_FOUND;

        TCB *tcb = it->second;
        while (tcb->state != TERMINATED)
        {
            currTCB->state = WAITING;
            currTCB->waitID = id;
            waiting_list.push_back(currTCB->id);
            thd_yield();
        }

        if(value)
            *value = tcb->ret_value;
            
        all_threads.erase(it);
        delete tcb;

        return WAIT_SUCCESSFUL;
	}

	/*!*****************************************************************************
		\brief Give the cpu to another thread context to run
	*******************************************************************************/
	void thd_yield(void)
	{
		// saving the context of the current thread! 
		asm volatile(
			"movq %%rbx, %0 \n"
			"movq %%rdi, %1 \n"
			"movq %%rsi, %2 \n"
			"movq %%r12, %3 \n"
			"movq %%r13, %4 \n"
			"movq %%r14, %5 \n"
			"movq %%r15, %6 \n"
			: 	"=m"(currTCB->gregs[REG_RBX]),
				"=m"(currTCB->gregs[REG_RDI]),
				"=m"(currTCB->gregs[REG_RSI]),
				"=m"(currTCB->gregs[REG_R12]),
				"=m"(currTCB->gregs[REG_R13]),
				"=m"(currTCB->gregs[REG_R14]),
				"=m"(currTCB->gregs[REG_R15]));

		asm volatile(
			"movq %%rax, %0 \n"
			"movq %%rdx, %1 \n"
			"movq %%rcx, %2 \n"
			"movq %%r8, %3 \n"
			"movq %%r9, %4 \n"
			"movq %%r10, %5 \n"
			"movq %%r11, %6 \n"
			: 	"=m"(currTCB->gregs[REG_RAX]),
				"=m"(currTCB->gregs[REG_RDX]),
				"=m"(currTCB->gregs[REG_RCX]),
				"=m"(currTCB->gregs[REG_R8]),
				"=m"(currTCB->gregs[REG_R9]),
				"=m"(currTCB->gregs[REG_R10]),
				"=m"(currTCB->gregs[REG_R11]));

		// switch stack (save all values on stack)
		asm volatile(
			"movq %%rsp, %0 \n"
			"movq %%rbp, %1 \n"
			: 	"=m"(currTCB->gregs[REG_RSP]),
				"=m"(currTCB->gregs[REG_RBP]));
		asm volatile(
			"pushfq \n"
			"popq %0 \n"
			: "=m"(currTCB->gregs[REG_EFL])
			:
			: "rsp");

		// check if ready_queue and new_stack is empty (do nth if both are empty)
		bool const REMPTY = ready_queue.empty(), NEMPTY = new_stack.empty();
		if (REMPTY && NEMPTY)
			return;

		// only if the current thread is running then i add it to ready queue
		if (currTCB->state == RUNNING)
		{
			currTCB->state = READY;
			ready_queue.push_back(currTCB->id);
		}

		if (!REMPTY)
		{
			// Process the thread at the front of ready_queue
			currTCB = all_threads[ready_queue.front()];
			ready_queue.pop_front();
			currTCB->state = RUNNING;

			asm volatile(
				"pushq %0 \n"
				"popfq \n\t"
				:
				: "m"(currTCB->gregs[REG_EFL])
				: "cc", "rsp");

			asm volatile(
				"movq %0,  %%rsp \n"
				"movq %1,  %%rbp\n"
				:
				: "m"(currTCB->gregs[REG_RSP]),
				  "m"(currTCB->gregs[REG_RBP]));

			// load the rest registers
			asm volatile(
				"movq %0, %%rbx\n"
				"movq %1, %%rdi \n"
				"movq %2, %%rsi \n"
				"movq %3, %%r12 \n"
				"movq %4, %%r13 \n"
				"movq %5, %%r14 \n"
				"movq %6, %%r15 \n"
				:
				: "m"(currTCB->gregs[REG_RBX]),
				  "m"(currTCB->gregs[REG_RDI]),
				  "m"(currTCB->gregs[REG_RSI]),
				  "m"(currTCB->gregs[REG_R12]),
				  "m"(currTCB->gregs[REG_R13]),
				  "m"(currTCB->gregs[REG_R14]),
				  "m"(currTCB->gregs[REG_R15]));
			asm volatile(
				"movq %0, %%rax \n"
				"movq %1, %%rcx \n"
				"movq %2, %%rdx \n"
				"movq %3, %%r8 \n"
				"movq %4, %%r9 \n"
				"movq %5, %%r10 \n"
				"movq %6, %%r11 \n"
				:
				: "m"(currTCB->gregs[REG_RAX]),
				  "m"(currTCB->gregs[REG_RCX]),
				  "m"(currTCB->gregs[REG_RDX]),
				  "m"(currTCB->gregs[REG_R8]),
				  "m"(currTCB->gregs[REG_R9]),
				  "m"(currTCB->gregs[REG_R10]),
				  "m"(currTCB->gregs[REG_R11]));
		}
		else
		{
			// Get the thread at the back
			currTCB = all_threads[new_stack.back()];
			new_stack.pop_back();
			currTCB->state = ThreadState::RUNNING;

			asm volatile(
				"pushq %0 \n"
				"popfq \n\t"
				:
				: "m"(currTCB->gregs[REG_EFL])
				: "cc", "rsp");
			asm volatile(
				"movq %0,  %%rsp \n"
				"movq %1,  %%rbp\n"
				:
				: "m"(currTCB->gregs[REG_RSP]),
				  "m"(currTCB->gregs[REG_RBP]));
			RunFunction();
		}
	}
}