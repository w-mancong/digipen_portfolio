#include <iostream>
#include <type_traits>

template <typename T1, typename T2>
auto modulo_plain(T1 n, T2 d)
{
	return n % d;
}

void test_modulo_plain()
{
	std::cout << "In function [test_modulo_plain]" << std::endl;
	int n = 9;
	int d = 5;

	std::cout << n << " = " << modulo_plain(n, d)
		<< " (mod " << d << ")" << std::endl;
}

///////////////////////////////////////////////////////
// SFINAE using type traits
template <typename T1, typename T2,
		  typename = std::enable_if_t< std::is_integral_v<T1> && std::is_integral_v<T2> > > // Confine T1 and T2 to be integral types
auto modulo_trait(T1 n, T2 d)
{
	return n % d;
}

void test_modulo_trait()
{
	std::cout << "In function [test_modulo_trait]" << std::endl;
	int n = 9;
	int d = 5;

	std::cout << n << " = " << modulo_trait(n, d)
		<< " (mod " << d << ")" << std::endl;
}

///////////////////////////////////////////////////////
template <typename T1, typename T2>
auto modulo_requires(T1 n, T2 d)
	requires requires{ n % d; }
{
	return n % d;
}

void test_modulo_requires()
{
	std::cout << "In function [test_modulo_requires]" << std::endl;
	int n = 9;
	int d = 5;

	std::cout << n << " = " << modulo_requires(n, d)
		<< " (mod " << d << ")" << std::endl;
}

///////////////////////////////////////////////////////
template <typename T1, typename T2>
concept ModuloSupport = requires (T1 n, T2 d) { n % d; };

template <typename T1, typename T2>
auto modulo_concepts(T1 n, T2 d) requires ModuloSupport<T1, T2>
{
	return n % d;
}

void test_modulo_concepts()
{
	std::cout << "In function [test_modulo_concepts]" << std::endl;
	int n = 9;
	int d = 5;

	std::cout << n << " = " << modulo_concepts(n, d)
		<< " (mod " << d << ")" << std::endl;
}

///////////////////////////////////////////////////////
template <typename T>
concept IntegralType = std::is_integral_v<T>;

template <typename T1, typename T2>
	requires IntegralType<T1> && IntegralType<T2>
auto modulo_constraints(T1 n, T2 d)
{
	return n % d;
}

void test_modulo_constraints()
{
	std::cout << "In function [test_modulo_constraints]" << std::endl;
	int n = 9;
	int d = 5;

	std::cout << n << " = " << modulo_constraints(n, d)
		<< " (mod " << d << ")" << std::endl;
}

int main(void)
{
	//test_modulo_plain();
	//test_modulo_trait();
	//test_modulo_requires();
	//test_modulo_concepts();
	test_modulo_constraints();
}