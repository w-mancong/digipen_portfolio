#include <shared.h>

namespace
{
	static char constexpr const* SERVER_IP = "127.0.0.1";
}

int main(void)
{
	Network::StartUpNetwork();
	Network::ConnectionInfo client{};

	Network::SetUpClient(client, SERVER_IP);
	
	// Write out to that socket
	Shared::Position pos{5.0f, 4.0f};
	Shared::Data d(pos, Shared::Type::Position);

	Network::SetMessage( client.buf, d );
	Network::SendMessageTo(client);

	Network::ReceiveMessage(client);
	d = Shared::Convert(client.buf);
	pos = Shared::Transform<Shared::Position>(d.ptr);

	std::cout << pos.x << " " << pos.y << std::endl;

	Network::CleanUpNetwork(client.s);
}