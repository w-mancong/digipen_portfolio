//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "compute_least_square.h"
#include "clock.h"
#include <iostream>

#ifndef _WIN32

#include <sched.h>
#endif
typedef double T;

T poly(T a[], T x, long degree);
T polyh(T a[], T x, long degree);
T poly_opt(T a[], T x, long degree);
//double get_clock_speed();
using namespace std;

#define test_func poly_opt

double polyh(double a[], double x, long degree) // polyh
{
	double result = a[degree];
	for (long i = degree - 1; i >= 0; i--)
		result = a[i] + x*result;
	return result;
}

double poly(double a[], double x, long degree)
{
    long i;
    double result = a[0];
    double xpwr = x;
    for(i=1; i <= degree; ++i)
    {
		result += a[i] * xpwr;
		xpwr = x * xpwr;
    }
    return result;
}


int main(int argc, char **argv)
{
    long i;
    int max_size, step_size;
    double min;//minimum to pass
#ifdef DEBUG
    fprintf(stderr, "%f\n", get_clock_speed());       
#endif
#ifdef _WIN32
    if (!SetProcessAffinityMask(GetCurrentProcess(), 0x1)) {
		fprintf(stderr, "%d\n", GetLastError());
		system("pause");
		return -1;
    }	
#else
	cpu_set_t mask; /* processors 0 */
	CPU_ZERO(&mask);
	CPU_SET(1, &mask);
	unsigned int len = sizeof(mask);
	if (sched_setaffinity(0, len, &mask) < 0) {
		perror("sched_setaffinity");
	}
#endif


    srand(time(NULL));
#if 1    
    //cin >> max_size;
    //cin >> step_size;
    //cin >> min;
	int largest_size;
	cout << "Input largest element size (Iterates in groups of 10): ";
    cin >> largest_size;
	step_size = 1;
	min = 0;

	if (largest_size <= 10)
	{
		cout << "Input must > 10!\n";
		return 0;
	}
	if (largest_size % 10 != 0)
	{
		cout << "Input must be iteration of 10!\n";
		return 0;
	}
#else    
    if (argc != 3) {
		fprintf(stderr, "Usage: %s <max-size> <step-size>\n", argv[0]);
		return 1;
    }
    max_size = atoi(argv[1]);
    step_size = atoi(argv[2]);
#endif
	cout << "========================================================\n";
	cout << "Function [Elements]: Cycles\n";
	cout << "========================================================\n";
	// poly
	for (max_size = largest_size/10; max_size <= largest_size; max_size += largest_size/10)
    {
		T *coeffs = (T *) malloc(sizeof(T) * max_size);
		T x;
		long j;
		volatile T unopt_results, opt_results;
		struct L_square_struct unopt_slope, opt_slope;

		init_l_square_struct(&unopt_slope);
		init_l_square_struct(&opt_slope);


		for (j = 0; j < max_size; ++j) {
			coeffs[j] = (T) (rand() % 5 + 1);
		}
		x = coeffs[0];

		prof_time_t cout_time;
		for (i = step_size; i < max_size; i += step_size) {
			volatile prof_time_t best_time, unopt_time, opt_time;
			volatile unsigned int s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l;
			int times;

			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				unopt_results = poly(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				unopt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (unopt_time < best_time)
					best_time = unopt_time;
			}
			unopt_time = best_time;
			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				opt_results = poly(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				opt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (opt_time < best_time)
					best_time = opt_time;
			}
			opt_time = best_time;
			cout_time = best_time;
			update(&unopt_slope, i, unopt_time);
			update(&opt_slope, i, opt_time);
#ifdef DEBUG			
			printf("%lu %llu %llu\n", i, unopt_time, opt_time);
#endif			
			if (fabs(unopt_results - opt_results)
					>= 0.0001 * fabs(unopt_results >
					 opt_results ? unopt_results :
					 opt_results)) {
				int k;
				fprintf(stderr, "something is wrong\n");
				fprintf(stderr, "%lu\n", i);
				fprintf(stderr, "unopt_results: %lf", unopt_results);
				fprintf(stderr, "opt_results: %lf", opt_results);

				//std::cerr << "i: " << i << std::endl;
				//std::cerr << "unopt_results: " << unopt_results << std::endl;
				//std::cerr << "opt_results: " << opt_results << std::endl;             
				//std::cerr << std::endl;

				for (k = 0; k < i; ++k) {
					fprintf(stderr, "coeffes[%d]=%lf\n", k, coeffs[k]);
					//  std::cerr << "coeffs[" << k << "]=" << coeffs[k] << std::endl;
				}
				exit(-1);
			}
		}
#ifdef DEBUG
		printf("Unopt CPE: %lf\n", get_current_slope(&unopt_slope));
		printf("Opt CPE: %lf\n", get_current_slope(&opt_slope));
#else
        double slope1 = get_current_slope(&unopt_slope);
        double slope2 = get_current_slope(&opt_slope);
        double ratio = slope1/slope2;
        if ( ratio > min)
            cout << "poly [" << max_size << " elem]: " << cout_time << "\n";    
        else
            cout << "fail " << ratio;
#endif		
		free(coeffs);
    }
	cout << "========================================================\n";
	
	// polyh
	for (max_size = largest_size/10; max_size <= largest_size; max_size += largest_size/10)
    {
		T *coeffs = (T *) malloc(sizeof(T) * max_size);
		T x;
		long j;
		volatile T unopt_results, opt_results;
		struct L_square_struct unopt_slope, opt_slope;

		init_l_square_struct(&unopt_slope);
		init_l_square_struct(&opt_slope);


		for (j = 0; j < max_size; ++j) {
			coeffs[j] = (T) (rand() % 5 + 1);
		}
		x = coeffs[0];

		prof_time_t cout_time;
		for (i = step_size; i < max_size; i += step_size) {
			volatile prof_time_t best_time, unopt_time, opt_time;
			volatile unsigned int s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l;
			int times;

			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				unopt_results = poly(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				unopt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (unopt_time < best_time)
					best_time = unopt_time;
			}
			unopt_time = best_time;
			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				opt_results = polyh(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				opt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (opt_time < best_time)
					best_time = opt_time;
			}
			opt_time = best_time;
			cout_time = best_time;
			update(&unopt_slope, i, unopt_time);
			update(&opt_slope, i, opt_time);
#ifdef DEBUG			
			printf("%lu %llu %llu\n", i, unopt_time, opt_time);
#endif			
			if (fabs(unopt_results - opt_results)
					>= 0.0001 * fabs(unopt_results >
					 opt_results ? unopt_results :
					 opt_results)) {
				int k;
				fprintf(stderr, "something is wrong\n");
				fprintf(stderr, "%lu\n", i);
				fprintf(stderr, "unopt_results: %lf", unopt_results);
				fprintf(stderr, "opt_results: %lf", opt_results);

				//std::cerr << "i: " << i << std::endl;
				//std::cerr << "unopt_results: " << unopt_results << std::endl;
				//std::cerr << "opt_results: " << opt_results << std::endl;             
				//std::cerr << std::endl;

				for (k = 0; k < i; ++k) {
					fprintf(stderr, "coeffes[%d]=%lf\n", k, coeffs[k]);
					//  std::cerr << "coeffs[" << k << "]=" << coeffs[k] << std::endl;
				}
				exit(-1);
			}
		}
#ifdef DEBUG
		printf("Unopt CPE: %lf\n", get_current_slope(&unopt_slope));
		printf("Opt CPE: %lf\n", get_current_slope(&opt_slope));
#else
        double slope1 = get_current_slope(&unopt_slope);
        double slope2 = get_current_slope(&opt_slope);
        double ratio = slope1/slope2;
        if ( ratio > min)
            cout << "polyh [" << max_size << " elem]: " << cout_time << "\n";    
        else
            cout << "fail " << ratio;
#endif		
		free(coeffs);
    }
	cout << "========================================================\n";

	//	opt_poly
	for (max_size = largest_size/10; max_size <= largest_size; max_size += largest_size/10)
    {
		T *coeffs = (T *) malloc(sizeof(T) * max_size);
		T x;
		long j;
		volatile T unopt_results, opt_results;
		struct L_square_struct unopt_slope, opt_slope;

		init_l_square_struct(&unopt_slope);
		init_l_square_struct(&opt_slope);


		for (j = 0; j < max_size; ++j) {
			coeffs[j] = (T) (rand() % 5 + 1);
		}
		x = coeffs[0];

		prof_time_t cout_time;
		for (i = step_size; i < max_size; i += step_size) {
			volatile prof_time_t best_time, unopt_time, opt_time;
			volatile unsigned int s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l;
			int times;

			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				unopt_results = poly(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				unopt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (unopt_time < best_time)
					best_time = unopt_time;
			}
			unopt_time = best_time;
			best_time = 0xFFFFFFFF;

			for (times = 0; times < 100; ++times) {
				start_read(&s_cyc_h, &s_cyc_l);
				opt_results = test_func(coeffs, x, i);
				end_read(&e_cyc_h, &e_cyc_l);
				opt_time =
					compute_time(s_cyc_h, s_cyc_l, e_cyc_h, e_cyc_l);
				if (opt_time < best_time)
					best_time = opt_time;
			}
			opt_time = best_time;
			cout_time = best_time;
			update(&unopt_slope, i, unopt_time);
			update(&opt_slope, i, opt_time);
#ifdef DEBUG			
			printf("%lu %llu %llu\n", i, unopt_time, opt_time);
#endif			
			if (fabs(unopt_results - opt_results)
					>= 0.0001 * fabs(unopt_results >
					 opt_results ? unopt_results :
					 opt_results)) {
				int k;
				fprintf(stderr, "something is wrong\n");
				fprintf(stderr, "%lu\n", i);
				fprintf(stderr, "unopt_results: %lf", unopt_results);
				fprintf(stderr, "opt_results: %lf", opt_results);

				//std::cerr << "i: " << i << std::endl;
				//std::cerr << "unopt_results: " << unopt_results << std::endl;
				//std::cerr << "opt_results: " << opt_results << std::endl;             
				//std::cerr << std::endl;

				for (k = 0; k < i; ++k) {
					fprintf(stderr, "coeffes[%d]=%lf\n", k, coeffs[k]);
					//  std::cerr << "coeffs[" << k << "]=" << coeffs[k] << std::endl;
				}
				exit(-1);
			}
		}
#ifdef DEBUG
		printf("Unopt CPE: %lf\n", get_current_slope(&unopt_slope));
		printf("Opt CPE: %lf\n", get_current_slope(&opt_slope));
#else
        double slope1 = get_current_slope(&unopt_slope);
        double slope2 = get_current_slope(&opt_slope);
        double ratio = slope1/slope2;
        if ( ratio > min)
            cout << "opt_poly [" << max_size << " elem]: " << cout_time << "\n";    
        else
            cout << "fail " << ratio;
#endif		
		free(coeffs);
    }
	cout << "========================================================\n";
    
#ifdef _WIN32
    system("pause");
#endif
}
