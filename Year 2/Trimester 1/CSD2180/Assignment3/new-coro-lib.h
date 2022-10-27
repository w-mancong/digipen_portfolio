#ifndef NEW_CORO_LIB_H
#define NEW_CORO_LIB_H

namespace CORO
{
using ThreadID = unsigned;
void thd_init();
ThreadID new_thd(void*(*thd_function_t)(void*), void * param);
void thread_exit(void *ret_value);
int wait_thread(ThreadID id, void **value);
void thd_yield();
const int WAIT_SUCCESSFUL = 0;
const int NO_THREAD_FOUND = -1;
enum ThreadState : int;
}

#endif