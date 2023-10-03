#include <iostream>
#include <stdint.h>

enum class ReflectType : int64_t
{
#define REFLECT_TYPE(type, type_name) r_##type_name,
#define REFLECT_CUSTOM_CTYPE(type, type_name) r_##type_name,
#define REFLECT_CUSTOM_STYPE(type, type_name) r_##type_name,

	Invalid = -1,
#include "ReflectType.def"
	Total,

#undef REFLECT_TYPE
#undef REFLECT_CUSTOM_CTYPE
#undef REFLECT_CUSTOM_STYPE
};

struct ReflectMembers
{
	char const* name = "";
	ReflectType type{ ReflectType::Invalid };
	size_t size{};
};

template<typename T>
ReflectType constexpr GetReflectType()
{
	return ReflectType::Invalid;
}

// To forward declare our custom types
#define REFLECT_CUSTOM_CTYPE(type, type_name)\
class type;

#define REFLECT_CUSTOM_STYPE(type, type_name)\
struct type;

#define REFLECT_TYPE(type, type_name)\

#include "ReflectType.def"
#undef REFLECT_TYPE
#undef REFLECT_CUSTOM_CTYPE
#undef REFLECT_CUSTOM_STYPE

#define REFLECT_TYPE(type, type_name)\
template<>\
ReflectType constexpr GetReflectType<type>()\
{\
	return ReflectType::r_##type_name;\
}\

#define REFLECT_CUSTOM_CTYPE(type, type_name)\
template<>\
ReflectType constexpr GetReflectType<type>()\
{\
	return ReflectType::r_##type_name;\
}\

#define REFLECT_CUSTOM_STYPE(type, type_name)\
template<>\
ReflectType constexpr GetReflectType<type>()\
{\
	return ReflectType::r_##type_name;\
}\

#include "ReflectType.def"
#undef REFLECT_TYPE
#undef REFLECT_CUSTOM_CTYPE
#undef REFLECT_CUSTOM_STYPE

std::ostream& operator<<(std::ostream& os, ReflectType type)
{
	return os << static_cast<int64_t>(type);
}

int main(void)
{
	ReflectType constexpr type = GetReflectType<Vector3>();
	std::cout << type << std::endl;
}