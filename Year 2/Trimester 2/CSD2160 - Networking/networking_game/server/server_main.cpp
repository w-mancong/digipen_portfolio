#include <shared.h>

int main(void)
{
	// Startup Winsock
	Network::StartUpNetwork();
	
	// Bind socket to ip address and port
	Network::ConnectionInfo server{};
	Network::SetUpServer(server.s);

	// Enter a loop
	while (true)
	{
		// Wait for message
		Network::ReceiveMessage(server);
		Shared::Position pos{};
		{
			Shared::Data d = Shared::Convert(server.buf);
			pos = Shared::Transform<Shared::Position>(d.ptr);
		}

		// Display message and client info
		char clientIp[Shared::BUFFER_SIZE];
		Network::ObtainIpAddr(server.addr, clientIp);

		std::cout << "Message received from " << clientIp << " : " << pos.x << " " << pos.y << std::endl;
		pos = { 10.0f, 15.0f };

		{
			Shared::Data d(pos, Shared::Type::Position);
			Network::SetMessage(server.buf, d);
		}

		Network::SendMessageTo(server);
	}

	Network::CleanUpNetwork(server.s);
}