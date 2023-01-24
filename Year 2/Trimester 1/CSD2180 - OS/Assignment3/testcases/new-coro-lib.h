#ifndef NEW_CORO_LIB_H
#define NEW_CORO_LIB_H

#include <cstdint>

namespace CORO
{
    using s8  = int8_t;
    using u8  = uint8_t;
    using s16 = int16_t;
    using u16 = uint16_t;
    using s32 = int32_t;
    using u32 = uint32_t;
    using s64 = int64_t;
    using u64 = uint64_t;

    using ThreadID = u32;
    s32 constexpr WAIT_SUCCESSFUL{ 0 };
    s32 constexpr NO_THREAD_FOUND{ -1 };
    enum class ThreadState : s32;

    void thd_init();
    ThreadID new_thd(void* (*thd_function_t)(void*), void * param);
    void thread_exit(void *ret_value);
    s32 wait_thread(ThreadID id, void **value);
    void thd_yield();
}

#endif