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

template < typename Func = void(), typename = std::is_function<Func> >
class EventDispatcher : public Functional
{
public:
	EventDispatcher() = default;
	EventDispatcher(std::function<Func> f) : func{ f } {};
	EventDispatcher(Func f) : func{ f } {};

	template<typename... Args>
	void operator()(Args... args) const
	{
		func(args...);
	}

private:
	std::function<Func> const func;
};

enum class EventType
{
#define EVENT_TYPE(event_name, function_type) event_name,
	Invalid = -1,
#include "Events.def"
	Total,
#undef EVENT_TYPE
};

template <unsigned>
struct GetDispatcherType {};

#define EVENT_TYPE(event_name, function_type)\
template<>\
struct GetDispatcherType<static_cast<unsigned>( EventType::##event_name )>\
{\
	static EventDispatcher<function_type> Get() { return {}; }\
};
#include "Events.def"
#undef EVENT_TYPE

class EventSystem
{
public:
	static EventSystem* GetInstance()
	{
		static EventSystem instance{};
		return &instance;
	}

	template <typename Func, typename = std::is_function<Func>>
	int AddListener(EventType eventType, Func f)
	{
		static int count = -1;
		m_events[eventType].emplace_back( std::make_pair( ++count, std::make_shared< EventDispatcher<Func> >(f) ) );
		return count;
	}

	void RemoveListener(EventType eventType, int index)
	{
		std::vector<Listeners>& v = m_events[eventType];
		v.erase(std::remove_if(
			v.begin(),
			v.end(),
			[=](Listeners x)
			{
				return index == x.first;
			}));
	}

	template <typename Func, typename... Args>
	void InvokeEvent(EventType eventType, Args... args)
	{
		auto it = m_events[eventType].begin();  auto const end = m_events[eventType].end();
		while(it != end)
		{
			auto func = *dynamic_cast<Func*>( it->second.get() );
			func(args...); ++it;
		};
	}

private:
	EventSystem()  = default;
	~EventSystem() = default;

	using Listeners = std::pair<int, std::shared_ptr<Functional>>;
	using Events = std::unordered_map< EventType, std::vector< Listeners > >;
	Events m_events{};
};

#define ADD_LISTENER(event_name, func_name) EventSystem::GetInstance()->AddListener<decltype(func_name)>(event_name, func_name)
#define REMOVE_LISTENER(event_name, index)  EventSystem::GetInstance()->RemoveListener(event_name, index)
#define INVOKE_EVENT(event_name, ...)\
EventSystem::GetInstance()->InvokeEvent< decltype( GetDispatcherType< static_cast<unsigned>(event_name) >::Get() ) >(event_name, __VA_ARGS__)

void Test0(int a)
{
	std::cout << "I am Test0. Argument value a is: " << a << std::endl;
}

void Test1(int a)
{
	std::cout << "I am Test1. Argument value a is: " << a << std::endl;
}

void Test2(int a, std::string const& str)
{
	std::cout << "I am Test2. Argument value a is: " << a << ' ' << str << std::endl;
}

int main()
{
	int a = ADD_LISTENER(EventType::MouseInput, Test0);
	int b = ADD_LISTENER(EventType::MouseInput, Test1);
	int c = ADD_LISTENER(EventType::KeyboardInput, Test2);
	//REMOVE_LISTENER(EventType::MouseInput, a);
	REMOVE_LISTENER(EventType::MouseInput, b);
	INVOKE_EVENT(EventType::MouseInput, 2);
	INVOKE_EVENT(EventType::KeyboardInput, 123, "Hello World");
}