#include "Reflection.h"

class Test
{
public:
	float a;
};

class Vector3
{
public:
	float x{}, y{}, z{};
	Test t{};

	ENABLE_REFLECTION
};

BEGIN_REFLECTION(Vector3)
REFLECT_MEMBER(x)
REFLECT_MEMBER(y)
REFLECT_MEMBER(z)
REFLECT_MEMBER(t)
END_REFLECTION(Vector3)

template <typename T>
void PrintMembers(T const& type)
{
	ReflectClass const& reflection = T::Reflect();
	for (auto const& member : reflection.members)
		std::cout << member.variable_type << ": " << member.name << " size: " << member.size << std::endl;
	std::cout << std::endl;
}

template <typename... Args>
void Print()
{
	([]
	{
		std::cout << typeid(Args).name() << std::endl;
	} (), ...);
}

int main(void)
{
	PrintMembers( Vector3() );
	Print<float, int, float>();
}