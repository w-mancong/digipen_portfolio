#include "new-coro-lib.h"
#include <unordered_map>
#include <deque>
#include <algorithm>

namespace CORO
{
	enum class ThreadState
	{
		INVALID = -1,
		NEW,
		READY,
		RUNNING,
		WAITING,
		TERMINATED,
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
		REG_R8,
		REG_R9,
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
		void *(*thd_function_t)(void *){nullptr};
		void *param{nullptr}; // argument values
		void *ret_value{nullptr}; // return values
		ThreadState state{ThreadState::INVALID};
		void *gregs[(u64)Register::TOTAL_REGISTERS]{nullptr};
	};

	namespace
	{
		// list of all threads
		// list of all READY-state threads
		// list of all WAITING-state threads
		// list of all TERMINATED-state threads
		// list of all NEW-state threads
		// ... (Add as needed tbh, I only used arnd 3 lists)

		u64 constexpr ONE_MB{ 1'024'000 };
		static ThreadID id_counter{ 0 };
		TCB *currTCB{ nullptr };

		std::deque<TCB *> ready_queue{}, new_stack{}, waiting_list{};
		std::unordered_map<ThreadID, TCB *> all_threads{};

		void RunFunction(void)
		{
			if (currTCB->thd_function_t)
				thread_exit(currTCB->thd_function_t(currTCB->param));
		}
	}

	// init the primary thread
	void thd_init(void)
	{
		TCB *tcb = new TCB;
		tcb->id = id_counter++;
		tcb->state = ThreadState::RUNNING;

		all_threads[tcb->id] = tcb;
		currTCB = tcb;
	}

	// creates a new thread
	ThreadID new_thd(void *(*thd_function_t)(void *), void *param)
	{
		TCB *tcb = new TCB;
		tcb->id = id_counter++;
		tcb->thd_function_t = thd_function_t;
		tcb->param = param;
		tcb->ret_value = nullptr;
		tcb->state = ThreadState::NEW;
		tcb->gregs[(u64)Register::REG_RBP] = (char *)malloc(sizeof(char *));
		tcb->gregs[(u64)Register::REG_RSP] = (char *)malloc(ONE_MB) + ONE_MB;

		new_stack.push_back(tcb);
		all_threads[tcb->id] = tcb;
		return tcb->id;
	}

	// terminates thread and yields
	void thread_exit(void *ret_value)
	{			
		// Set the return value for the thread as ret_value (currThrd->ret_val = ret_value)
		currTCB->ret_value = ret_value;
		// Set the state of the thread to TERMINATED
		currTCB->state = ThreadState::TERMINATED;

		// Check if there are any threads waiting for this thread to terminate
		// if there are, set their values as required and put them in the READY list (If required, based on your implementation)
		auto it = std::find_if(waiting_list.begin(), waiting_list.end(), [](TCB* tcb)
		{
			return tcb->waitID == currTCB->id;
		});

		// Found a thread that is waiting for the current thread to be completed
		TCB *tcb = (*it);
		if(it != waiting_list.end())
		{
			tcb->state = ThreadState::READY;
			ready_queue.push_back(tcb);
			waiting_list.erase(it);
		}
		
		// yield CPU to another thread (calls thd_yield())
		thd_yield();
	}

	//  wait for another thread
	s32 wait_thread(ThreadID id, void **value)
	{
		auto it = all_threads.find(id);
		if (it == all_threads.end())
			return NO_THREAD_FOUND;

		TCB *tcb = it->second;
		while (tcb->state != ThreadState::TERMINATED)
		{
			currTCB->state = ThreadState::WAITING;
			currTCB->waitID = id;
			waiting_list.push_back(currTCB);

			thd_yield();
		}

		if (value)
			*value = tcb->ret_value;

		all_threads.erase(it);
		delete tcb;

		return WAIT_SUCCESSFUL;
	}

	// let the current thread yield CPU and let the next thread run
	void thd_yield(void)
	{
		/*
			Context Saving:
			Saving of all the registers
		*/
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
		asm volatile(
			"movq %%rsp, %0 \n"
			"movq %%rbp, %1 \n"
			: "=m"(currTCB->gregs[(u64)Register::REG_RSP]),
			  "=m"(currTCB->gregs[(u64)Register::REG_RBP]));

		// Find the next thread
		bool const READY_FILLED = ready_queue.size(), NEW_FILLED = new_stack.size();

		// Both the ready queue and new_stack is empty, do nth
		if (!READY_FILLED && !NEW_FILLED)
			return;

		// If current thread is running, add it to ready queue
		if (currTCB->state == ThreadState::RUNNING)
		{
			currTCB->state = ThreadState::READY;
			ready_queue.push_back(currTCB);
		}

		if (READY_FILLED)
		{
			currTCB = ready_queue.front(); ready_queue.pop_front();
			currTCB->state = ThreadState::RUNNING;

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
		}
		else if (NEW_FILLED)
		{
			currTCB = new_stack.back(); new_stack.pop_back();
			currTCB->state = ThreadState::RUNNING;
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
			RunFunction();
		}
	}
}