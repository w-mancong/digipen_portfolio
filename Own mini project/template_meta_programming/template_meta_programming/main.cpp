#include <iostream>
#include <type_traits>
#include <utility>

template <typename T>
struct BoxedType{};

// an instance of BoxedType<T>
template <typename T>
constexpr auto BoxedInstance = BoxedType<T>{};

// function without function body
template <typename T>
constexpr T StripBoxedType(BoxedType<T>);

// SFINAE out using this function without function body
template<typename LambdaType, typename... ArgsTypes,
	typename = decltype( std::declval<LambdaType>()(std::declval<ArgsTypes&&>()...) )> 
	std::true_type is_implemented(void*);

// catch all function without function body
template<typename LambdaType, typename... ArgTypes>
std::false_type is_implemented(...);

constexpr auto is_implementation_valid = [](auto lambda_instance)
{
	return [](auto&&... lambda_args)
	{
		return decltype(is_implemented<decltype(lambda_instance), decltype(lambda_args)&&...>(nullptr)){};
	};
};

constexpr auto is_default_constructible_lambda = [](auto boxed_instance) -> decltype( decltype( StripBoxedType(boxed_instance) )() ) {};

constexpr auto is_default_construtible_helper = is_implementation_valid(is_default_constructible_lambda);

#define TYPE_NAME(type) typeid(type).name()

int main(void)
{
	std::cout << "Is " << TYPE_NAME(int) << " default constructible? "
		<< std::boolalpha << (is_default_construtible_helper(BoxedInstance<int&>) ? "yes" : "no") <<  std::endl;
}