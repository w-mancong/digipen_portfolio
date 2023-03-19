/*!
file:	shared.h
author:	Wong Man Cong
email:	w.mancong\@digipen.edu
brief:	This file contains function definition that will be used for both server and clients

		All content © 2023 DigiPen Institute of Technology Singapore. All rights reserved.
*//*__________________________________________________________________________________*/
#ifndef	SHARED_H
#define SHARED_H

#include <iostream>
#include <WS2tcpip.h>

#pragma comment (lib, "ws2_32.lib")

namespace Shared
{
	static size_t constexpr const BUFFER_SIZE = 1024;

	/*!*********************************************************************************
		\brief Will be used to keep track of the type of data be send/received
	***********************************************************************************/
	enum class Type : uint64_t
	{
		Invalid = 0,
		Position,
	};

	/*!*********************************************************************************
		\brief Generic struct that contains all the different type of data ready to
			   be sent
	***********************************************************************************/
	struct Data
	{
		Type t{ Type::Invalid };
		char ptr[BUFFER_SIZE - sizeof(Type)]{};

		template <typename T>
		Data(T const& val, Type type = Type::Invalid)
		{
			Set(val, type);
		}

		template <typename T>
		void Set(T const& val, Type type = Type::Invalid)
		{
			t = type;
			ZeroMemory(ptr, sizeof(ptr));
			memcpy_s(ptr, sizeof(ptr), &val, sizeof(T));
		}
	};

	/*!*********************************************************************************
		\brief Position struct
	***********************************************************************************/
	struct Position
	{
		float x{}, y{};
	};

	/*!*********************************************************************************
		\brief Converts char buffer to the generic "Data" object

		\param [in] buf: Message that is set using the SetMessage function ready to be 
						 converted into the generic "Data" object

		\return Converted message from char const* to the generic "Data" object
	***********************************************************************************/
	Data Convert(char const* buf)
	{
		return *reinterpret_cast<Data*>( const_cast<char*>(buf) );
	}

	/*!*********************************************************************************
		\brief Converts the generic "Data" object to a pointer to a char const

		\param [in] d: A generic "Data" object used to store information to be send

		\return Pointer to a char const to the head of the address of d
	***********************************************************************************/
	char const* Convert(Data& d)
	{
		return reinterpret_cast<char const*>(&d);
	}

	/*!*********************************************************************************
		\brief Transform the address containing the value to the specific object

		\param [in] ptr: This parameter should be coming from Data::ptr after calling
						 the Convert function that converts the char buffer to the
						 "Data" object
	***********************************************************************************/
	template <typename T>
	T Transform(void* ptr)
	{
		return *reinterpret_cast<T*>(ptr);
	}
}

namespace Network
{
	/*!*********************************************************************************
		\brief Struture containing all the neccessary data for establishing a UDP connection
	***********************************************************************************/
	struct ConnectionInfo
	{
		SOCKET s{};							// SOCKET
		sockaddr_in addr{};					// contains infomation on who recv/send the message from
		char buf[Shared::BUFFER_SIZE]{};	// char buffer used to recv/send messages

		ConnectionInfo(void) : s{ socket(AF_INET, SOCK_DGRAM, 0) }
		{
			ZeroMemory( &addr, sizeof(addr) );
			ZeroMemory( buf, sizeof(buf) );
		}
	};

	/*!*********************************************************************************
		\brief Start up WSA
	***********************************************************************************/
	void StartUpNetwork(void)
	{
		WSADATA data{};
		WORD version = MAKEWORD(2, 2);
		int wsOk = WSAStartup(version, &data);
		if (wsOk)
		{
			std::cout << "can't start Winsock! Error Code: " << wsOk;
			std::exit(EXIT_FAILURE);
		}
	}

	/*!*********************************************************************************
		\brief Set up client connection

		\param [in] client: Can just pass in a newly created ConnectionInfo
		\param [in] ipAddr: IP Address of the server
	***********************************************************************************/
	void SetUpClient(ConnectionInfo& client, char const* ipAddr)
	{
		// Create a hint structure for the server
		client.addr.sin_family = AF_INET;
		client.addr.sin_port = htons(54000);
		inet_pton(AF_INET, ipAddr, &client.addr.sin_addr);
	}

	/*!*********************************************************************************
		\brief Set up server connection

		\param [in] server: Socket used by server
	***********************************************************************************/
	void SetUpServer(SOCKET server)
	{
		sockaddr_in serverHint{};
		serverHint.sin_addr.S_un.S_addr = ADDR_ANY;
		serverHint.sin_family = AF_INET;
		serverHint.sin_port = htons(54000);	// Convert from little to big endian

		if (bind(server, (sockaddr*)&serverHint, sizeof(serverHint)) == SOCKET_ERROR)
		{
			std::cout << "Can't bind socket! Error Code: " << WSAGetLastError() << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	/*!*********************************************************************************
		\brief Closes socket s and cleans up WSA
	***********************************************************************************/
	void CleanUpNetwork(SOCKET s)
	{
		// Close socket
		closesocket(s);
		// Shutdown winsock
		WSACleanup();
	}

	/*!*********************************************************************************
		\brief Helper function to send a message

		\param [in] s: Socket established either by server/client
		\param [in] buf: Message to be sent
		\param [in] in: Contains information to the recipient
	***********************************************************************************/
	void SendMessageTo(SOCKET s, char const* buf, sockaddr_in const& in)
	{
		int sendOk = sendto(s, buf, Shared::BUFFER_SIZE, 0, (sockaddr*)&in, sizeof(in));
		if (sendOk == SOCKET_ERROR)
			std::cout << "Unable to send message. Error Code: " << WSAGetLastError() << std::endl;
	}

	/*!*********************************************************************************
		\brief Helper function to send a message

		\param [in] info: ConnectionInfo of either the server/client
	***********************************************************************************/
	void SendMessageTo(ConnectionInfo const& info)
	{
		SendMessageTo(info.s, info.buf, info.addr);
	}

	/*!*********************************************************************************
		\brief Helper function to receive a message

		\param [in] s: Socket established either by server/client
		\param [out] buf: Char buffer used to store the message received
		\param [in] from: Contains information of the sender 
	***********************************************************************************/
	void ReceiveMessage(SOCKET s, char* buf, sockaddr_in const& from)
	{
		ZeroMemory(buf, Shared::BUFFER_SIZE);
		int fromlen = static_cast<int>( sizeof(from) );

		int recvOk = recvfrom(s, buf, Shared::BUFFER_SIZE, 0, (sockaddr*)&from, &fromlen);
		if (recvOk == SOCKET_ERROR)
			std::cout << "Unable to receive message. Error Code: " << WSAGetLastError() << std::endl;
	}

	/*!*********************************************************************************
		\brief Helper function to receive a message

		\param [in, out] info: ConnectionInfo of either the server/client
	***********************************************************************************/
	void ReceiveMessage(ConnectionInfo& info)
	{
		ReceiveMessage(info.s, info.buf, info.addr);
	}

	/*!*********************************************************************************
		\brief Helper function to set a specific message

		\param [out] dst: Destination of where the message will be stored
		\param [in] src: Source of the message to be stored
	***********************************************************************************/
	void SetMessage(char* dst, char const* src)
	{
		ZeroMemory(dst, Shared::BUFFER_SIZE);
		memcpy_s( dst, Shared::BUFFER_SIZE, src, strlen(src) );
	}

	/*!*********************************************************************************
		\brief Helper function to convert generic "Data" object and storing it into
			   a char buffer

		\param [out] dst: Destination of where the data will be stored
		\param [in] d: Data containing the type and bytes that is being sent
	***********************************************************************************/
	void SetMessage(char* dst, Shared::Data const& d)
	{
		ZeroMemory(dst, Shared::BUFFER_SIZE);
		memcpy_s(dst, Shared::BUFFER_SIZE, &d, sizeof(d));
	}

	/*!*********************************************************************************
		\brief Helper function to retrieve the ip address

		\param [in] addr: Contains information of the ip address from the sender/recipient
		\param [out] buf: Char buffer to store the ip address
	***********************************************************************************/
	void ObtainIpAddr(sockaddr_in const& addr, char* buf)
	{
		ZeroMemory(buf, Shared::BUFFER_SIZE);
		inet_ntop(AF_INET, &addr.sin_addr, buf, Shared::BUFFER_SIZE);
	}
}

#endif