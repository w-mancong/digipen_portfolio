#ifndef CLOCK_H
#define CLOCK_H

#ifdef __GNUC__
#include <unistd.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#include <intrin.h>
#endif

typedef unsigned long long prof_time_t;

void __inline start_read(volatile unsigned int *cycles_high, volatile unsigned int *cycles_low)
{
    
#ifdef _WIN32
	int a[4], b;
	__int64 result;
	__cpuid(a, b);
	result=__rdtsc();
	*cycles_low=*((unsigned int*)&result);
	*cycles_high=*(((unsigned int*)&result)+1);
	/*		
		cpuid
		rdtsc
		mov cycles_high, edx
		mov cycles_low, eax
	*/
#elif defined(__GNUC__)
	int low, high;
    asm volatile ("CPUID\n\t"
                  "RDTSC\n\t"
                  "mov %%edx, %0\n\t"
                  "mov %%eax, %1\n\t":"=r" (high), "=r"(low)
                  ::"%rax", "%rbx", "%rcx", "%rdx");
    *cycles_high=high;
    *cycles_low=low;
#endif
};

void __inline end_read(volatile unsigned int *cycles_high, volatile unsigned int *cycles_low)
{
#ifdef _WIN32
	__int64 result;
	int a[4], b;
	result=__rdtscp(cycles_high);
	__cpuid(a, b);	
	*cycles_low=*((unsigned int*)&result);
	*cycles_high=*(((unsigned int*)&result)+1);	
#elif defined(__GNUC__)
    asm volatile ("RDTSCP\n\t"
                  "mov %%edx, %0\n\t"
                  "mov %%eax, %1\n\t"
                  "CPUID\n\t":"=r" (*cycles_high), "=r"(*cycles_low)
                  ::"%rax", "%rbx", "%rcx", "%rdx");
#endif				  
};

prof_time_t compute_time(unsigned int s_cyc_h,
                         unsigned int s_cyc_l,
                         unsigned int e_cyc_h, unsigned int e_cyc_l)
{
    prof_time_t start = 0, end = 0;
    start = (((prof_time_t) s_cyc_h) << 32) | s_cyc_l;
    end = (((prof_time_t) e_cyc_h) << 32) | e_cyc_l;
    if (end > start)
        return (end - start);
    else
        return start - end;

}


double __inline get_clock_speed()
{
    unsigned int s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l;

    start_read(&s_cyc_h, &s_cyc_l);
#ifdef __GNUC__
    sleep(1);
#endif
#ifdef _WIN32
	Sleep(1000);
#endif
    end_read(&e_cyc_h, &e_cyc_l);
    return compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
}

#endif