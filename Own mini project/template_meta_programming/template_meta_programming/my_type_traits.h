#pragma once

template <typename T>
struct is_integral
{
	static bool constexpr const value = false;
};

template<>
struct is_integral<signed char>
{
	static bool constexpr const value = true;
	using type = signed char;
};

template<>
struct is_integral<short>
{
	static bool constexpr const value = true;
	using type = short;
};

template<>
struct is_integral<int>
{
	static bool constexpr const value = true;
	using type = int;
};

template<>
struct is_integral<long long>
{
	static bool constexpr const value = true;
	using type = long long;
};

template<>
struct is_integral<unsigned char>
{
	static bool constexpr const value = true;
	using type = unsigned char;
};

template<>
struct is_integral<unsigned short>
{
	static bool constexpr const value = true;
	using type = unsigned short;
};

template<>
struct is_integral<unsigned int>
{
	static bool constexpr const value = true;
	using type = unsigned int;
};

template<>
struct is_integral<unsigned long long>
{
	static bool constexpr const value = true;
	using type = unsigned long long;
};

template <typename T>
bool constexpr const is_valid = is_integral<T>::value;

template <typename T>
using integral_type = typename is_integral<T>::type;

template <typename T1, typename T2>
struct st_same_t
{
	static bool constexpr const value = false;
};

template <typename T>
struct st_same_t<T, T>
{
	static bool constexpr const value = true;
	using type = T;
};

template <typename T1, typename T2>
bool constexpr const is_same_v = st_same_t<T1, T2>::value;

template <typename T1, typename T2>
using is_same_t = typename st_same_t<T1, T2>::type;

template <bool TRUTH, typename RType>
struct st_enable_true
{
	static bool constexpr const value = false;
};

template <typename RType>
struct st_enable_true<true, RType>
{
	static bool constexpr const value = true;
	using type = RType;
};

template <bool TRUTH, typename RType>
using enable_if = typename st_enable_true<TRUTH, RType>::type;