#include <iostream>
#include <Windows.h>
#include <functional>
#include <type_traits>
#include <vector>
#include <memory>
#include <unordered_map>

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
	void AddListener(EventType eventType, Func f)
	{
		m_events[eventType].emplace_back( std::make_shared< EventDispatcher<Func> >(f) );
	}

	template <typename Func, typename... Args>
	void InvokeEvent(EventType eventType, Args... args)
	{
		auto it = m_events[eventType].begin();  auto const end = m_events[eventType].end();
		do
		{
			auto func = *dynamic_cast<Func*>( it->get() );
			func(args...);

		} while ( ++it != end );
	}

private:
	EventSystem()  = default;
	~EventSystem() = default;

	using Events = std::unordered_map< EventType, std::vector< std::shared_ptr<Functional> > >;
	Events m_events{};
};

#define ADD_LISTENER(event_name, func_name) EventSystem::GetInstance()->AddListener<decltype(func_name)>(event_name, func_name)
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

int main()
{
	std::function<void(int)> a(Test0), b(Test1);	

	ADD_LISTENER(EventType::MouseInput, Test0);
	ADD_LISTENER(EventType::KeyboardInput, Test1);
	INVOKE_EVENT(EventType::MouseInput, 2);
	INVOKE_EVENT(EventType::KeyboardInput, 3);
}