#pragma once

#define SINGLETON_Pointer(Class)\
public:\
	static Class* GetInstance()\
	{\
		if(!instance)\
			instance = new Class{};\
		return instance;\
	}\
	static void DeleteInstance()\
	{\
		if(instance)\
		{\
			delete instance;\
			instance = nullptr;\
		}\
	}\
private:\
	Class()  = default;\
	~Class() = default;\
	static Class* instance;\

#define SINGLETON_Reference(Class)\
public:\
	static Class& GetInstance()\
	{\
		static Class instance{};\
		return instance;\
	}\
private:\
	Class()  = default;\
	~Class() = default;\

#define SINGLETON_Pointer_Init(Class)\
Class* Class::instance = nullptr;