struct CompleteMagazineAEvent
{
    MagazineA magazine;
}

struct MagazineA 
{
    int volume;
}


EventManager::SendEvent<CompleteMagazineA>(MagazineA{1});


Company A distributes Magazine A
-> completes magazine A volume 1 triggers CompleteMagazineEvent
-> 

Mailman  (dispatcher)

Subscribers:
House 1
House 2
House 3

// EventManager (Company)
// Event (Magazine)
// Dispatcher (Mailman)
// Listener (Houses)

template <typename ... TEventArgs>
class EventListener
{
    using listenerFunc = std::function<void(TEventArgs...)>

    std::vector<listenerFunc> listeners;

    void AddListener(listenerFunc f) {
        listeners.push_back(f);
    }

    void SendMessage(TEventArgs... args) {
        for (auto& i : listener) {
            listener(args...);
        }
    } 
};

template<typename ... TEventArgs>
class Message 
{
    std::string name;

    std::tuple<TEventArgs...> args;
};

template <typename .... TEventArgs>
class DispatcherManager : 
{
    using listenerFunc = std::function<void(TEventArgs...)>

    static std::unordered_map<std::string, EventListener<TEventArgs...>> m_Listeners{}
    
    // Used for sending messages in the future
    static std::vector<Message> messages;

    static void SendMessage(std::string name, TEventArgs... args) {
        auto iter = m_Listeners.find(name)

        if (iter != m_Listeners.end()) {
            m_Listeners.SendMessage(args...);
        }
    }

    static void AddListener(std::string name, listenerFunc f) {
        auto iter = m_Listeners.find(name)

        if (iter == m_Listeners.end()) {
            m_Listeners[name] = EventListener<TEventArgs...>{};
        }

        m_Listener[name].AddListener(f);
    }
};


class EventManager {
    
    template <typename ... TEventArgs>
    static void SendMessage(std::string name, TEventArgs ... args) {
        DispatcherManager<TEventArgs...>::SendMessage(name, args...);
    }

    template <typename ... TEventArgs>
    static void AddListener(std::string name, void(*listenerFunc)(TEventArgs...)) {
        DispatcherManager<TEventArgs...>::AddListener(name, listenerFunc);
    }
}