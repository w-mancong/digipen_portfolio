#pragma once

int fib(int x)
{
	if (x == 0 || x == 1)
		return x;
	return fib(x - 1) + fib(x - 2);
}

template <unsigned int N>
struct st_fib
{
	static const unsigned long long value = st_fib<N - 1>::value + st_fib<N - 2>::value;
};

template <>
struct st_fib<0>
{
	static const unsigned long long value = 0;
};

template <>
struct st_fib<1>
{
	static const unsigned long long value = 1;
};