#include "Reflection.h"

std::ostream& operator<<(std::ostream& os, ReflectType type)
{
	return os << static_cast<int64_t>(type);
}