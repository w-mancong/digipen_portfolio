#pragma once
#include <iostream>
#include <stdint.h>
#include <vector>
#include <string>
#include <cassert>

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

template<typename T>
ReflectType constexpr GetReflectType()
{
	return ReflectType::Invalid;
}

// To forward declare our custom types so that GetReflectType() can be defined here
#define REFLECT_CUSTOM_CTYPE(type, type_name)\
class type;
#define REFLECT_CUSTOM_STYPE(type, type_name)\
struct type;
#define REFLECT_TYPE(type, type_name)
#include "ReflectType.def"
#undef REFLECT_TYPE
#undef REFLECT_CUSTOM_CTYPE
#undef REFLECT_CUSTOM_STYPE

// Defining GetReflectType
#define GET_REFLECT_TYPE(type, type_name)\
template<>\
ReflectType constexpr GetReflectType<type>()\
{\
	return ReflectType::r_##type_name;\
}\

#define REFLECT_TYPE(type, type_name) GET_REFLECT_TYPE(type, type_name)
#define REFLECT_CUSTOM_CTYPE(type, type_name) GET_REFLECT_TYPE(type, type_name)
#define REFLECT_CUSTOM_STYPE(type, type_name) GET_REFLECT_TYPE(type, type_name)

#include "ReflectType.def"
#undef REFLECT_TYPE
#undef REFLECT_CUSTOM_CTYPE
#undef REFLECT_CUSTOM_STYPE
#undef GET_REFLECT_TYPE

std::ostream& operator<<(std::ostream& os, ReflectType type);

struct ReflectMembers
{
	char const* name = "";
	char const* variable_type = "";
	ReflectType type{ ReflectType::Invalid };
	size_t size{};
};

struct ReflectClass
{
	std::vector<ReflectMembers> members{};
	ReflectMembers const& operator[](size_t index)
	{
		assert(index < members.size());
		return members[index];
	}
	size_t size() const { return members.size(); }
};

template <typename T>
ReflectClass const& GetReflection();

#define BEGIN_REFLECTION(Class)\
template<>\
ReflectClass const& GetReflection<Class>()\
{\
	using ClassType = Class;\
	static ReflectClass reflection;\
	static bool init = false;\
	if(init) return reflection;\

#define REFLECT_MEMBER(Member)\
{\
	using MemberType = decltype(ClassType::Member); \
	reflection.members.emplace_back(ReflectMembers{ #Member, typeid(MemberType).name(), GetReflectType<MemberType>(), sizeof(MemberType) });\
}\

#define END_REFLECTION(Class)\
	init = true;\
	return reflection;\
}\
ReflectClass const& Class::Reflect()\
{\
	return GetReflection<Class>();\
}\

#define ENABLE_REFLECTION static ReflectClass const& Reflect();