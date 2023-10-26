#include <iostream>
#include <Windows.h>
#include <functional>
#include <type_traits>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

class Functional
{
public:
	Functional() = default;
	virtual ~Functional() = 0 {};
};

template < typename... Args >
class EventDispatcher : public Functional
{
public:
	EventDispatcher() = default;
	EventDispatcher(std::function<void(Args...)> f) : func{ f } {};

	template<typename... Args>
	void operator()(Args... args) const
	{
		func(args...);
	}

private:
	std::function<void(Args...)> func;
};

enum class EventType
{
#define EVENT_TYPE(event_name, ...) event_name,
	Invalid = -1,
#include "Events.def"
	Total,
#undef EVENT_TYPE
};

#pragma region Partial template specialization
// *************************************************************************************
template <unsigned>
struct GetType {};
#define EVENT_TYPE(event_name, ...)\
template<>\
struct GetType<static_cast<unsigned>( EventType::##event_name )>\
{\
	static EventDispatcher<__VA_ARGS__> GetEDT() { return {}; }\
	static std::function<void(__VA_ARGS__)> GetFn() { return {}; }\
};
#include "Events.def"
#undef EVENT_TYPE
#pragma endregion

class EventSystem
{
public:
	static EventSystem* GetInstance()
	{
		static EventSystem instance{};
		return &instance;
	}

	// EDT is EventDispatcher's type, func is the type of function
	template <typename EDT, typename Func, typename = std::is_function<Func>>
	int AddListener(EventType eventType, Func f)
	{
		static int count = -1;
		m_events[eventType].emplace_back( std::make_pair( ++count, std::make_shared< EDT >(f) ) );
		return count;
	}

	void RemoveListener(EventType eventType, int index)
	{
		std::vector<Listeners>& v = m_events[eventType];
		v.erase(std::remove_if(
			v.begin(),
			v.end(),
			[index](Listeners const& x)
			{
				return index == x.first;
			}));
	}

	template <typename EDT, typename... Args>
	void InvokeEvent(EventType eventType, Args... args)
	{
		auto it = m_events[eventType].begin();  auto const end = m_events[eventType].end();
		while(it != end)
		{
			/*
				If a run-time error occurs here, check if: 
				1) The total number of arguments are passed in correctly. 
				2) The argument types are correct

				Note: If you need to check, you can mouse over the enumeration
				of EventType, or simply check Events.def to see what the 
				type/total number of arguments for that specific event
			*/
			auto const func = *dynamic_cast<EDT*>( (it++)->second.get() );
			func(args...);
		};
	}

private:
	EventSystem()  = default;
	~EventSystem() = default;

	using Listeners = std::pair<int, std::shared_ptr<Functional>>;
	using Events = std::unordered_map< EventType, std::vector< Listeners > >;
	Events m_events{};
};

#define ADD_LISTENER(event_name, func_name)\
EventSystem::GetInstance()->AddListener< decltype(GetType<static_cast<unsigned>(event_name)>::GetEDT()),\
										 decltype(GetType<static_cast<unsigned>(event_name)>::GetFn())>(event_name, func_name)
#define REMOVE_LISTENER(event_name, index)\
EventSystem::GetInstance()->RemoveListener(event_name, index)
#define INVOKE_EVENT(event_name, ...)\
EventSystem::GetInstance()->InvokeEvent< decltype( GetType<static_cast<unsigned>(event_name)>::GetEDT())>(event_name, __VA_ARGS__)

#pragma region Example of how to use EventSystem
class A
{
public:
	A()
	{
		Init();
	}

	void Init()
	{
		ADD_LISTENER(EventType::MouseInput, [&](int a) { Test(a); });
	};

private:
	void Test(int a)
	{
		std::cout << "I am an class A object: " << a << std::endl;
	}
};

void Test0(int a)
{
	std::cout << "I am Test0 with an int parameter. Argument value a is: " << a << std::endl;
}

void Test1(int a)
{
	std::cout << "I am Test1 with an int parameter. Argument value a is: " << a << std::endl;
}

void Test2(int a, std::string const& str)
{
	std::cout << "I am Test2 with an int and string parameter. Argument value a is: " << a << ' ' << str << std::endl;
}

int main()
{
	A a1;
	int a = ADD_LISTENER(EventType::MouseInput, Test0);
	int b = ADD_LISTENER(EventType::MouseInput, Test1);
	int c = ADD_LISTENER(EventType::KeyboardInput, Test2);
	//REMOVE_LISTENER(EventType::MouseInput, a);
	REMOVE_LISTENER(EventType::MouseInput, b);
	INVOKE_EVENT(EventType::MouseInput, 2);
	INVOKE_EVENT(EventType::KeyboardInput, 123, "Hello World");
}
#pragma endregion