#include <WinSock2.h>
#include <iostream>
#include <string>
#include <unordered_map>

struct CLIENT_INFO
{
	SOCKET hClientSocket{};
	struct sockaddr_in clientAddr{};
	std::string userName{};
};

char szServerIPAddr[] = "127.0.0.1";	// Put here the IP address of the server
int nServerPort = 5050;					// The server port that will be used by
                                        // clients to talk with the server

bool InitWinSock2_0();
BOOL WINAPI ClientThread(LPVOID lpData);
void SendMessageToClient(char* pBuffer, CLIENT_INFO const* pClientInfo);
void SendMessageToAllClient(char* pBuffer);
void SetMessage(char* pBuffer, char const* message);
void SetNullTerminator(char* pBuffer, size_t max);
void wait(void);
void signal(void);

size_t constexpr BUFFER_SIZE = 1024;
std::unordered_map<std::string, CLIENT_INFO> users{};
std::atomic<int> s = 1; // semaphore

int main(void)
{
	if (!InitWinSock2_0())
	{
		std::cout << "Unable to Initialize Windows Socket environment" << WSAGetLastError() << std::endl;
		return -1;
	}

	// Create the server socket
	SOCKET hServerSocket;
	hServerSocket = socket(
		AF_INET,			// The address family. AF_INET specifies TCP/IP
		SOCK_STREAM,		// Protocol type. SOCKSTREAM specifies TCP
		0					// Protocol Name. Should be 0 for AF_INET address family
	);

	if (hServerSocket == INVALID_SOCKET)
	{
		std::cout << "Unable to create Server socket" << std::endl;
		// Cleanup the environment initialized by WSAStartup()
		WSACleanup();
		return -1;
	}

	// Create the structure describing various Server parameters
	struct sockaddr_in serverAddr;

	serverAddr.sin_family		= AF_INET;
	serverAddr.sin_addr.s_addr	= inet_addr(szServerIPAddr);
	serverAddr.sin_port			= htons(nServerPort);

	// Bind the Server socket to the address & port
	if ( bind( hServerSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr) ) == SOCKET_ERROR )
	{
		std::cout << "Unable to bind to " << szServerIPAddr << " port " << nServerPort << std::endl;
		// Free the socket and cleanup the environment initialized by WSAStartup()
		closesocket(hServerSocket);
		WSACleanup();
		return -1;
	}

	// Put the Server socket in listen state so that it can wait for client connections
	if ( listen(hServerSocket, SOMAXCONN) == SOCKET_ERROR )
	{
		std::cout << "Unable to put server in listen state" << std::endl;
		// Free the socket and cleanup the environment initialized by WSAStartup()
		closesocket(hServerSocket);
		WSACleanup();
		return -1;
	}

	// Start the infinite loop
	while (true)
	{
		/*
			As the socket is in listen mode there is a connection request pending.
			Calling accept() will succeed and return the socket for the request.
		*/
		SOCKET hClientSocket;
		struct sockaddr_in clientAddr;
		int nSize = sizeof(clientAddr);

		hClientSocket = accept(hServerSocket, (struct sockaddr*)&clientAddr, &nSize);
		if (hClientSocket == INVALID_SOCKET)
			std::cout << "accept() failed" << std::endl;
		else
		{	// Client thread creation
			HANDLE hClientThread;
			struct CLIENT_INFO clientInfo;
			DWORD dwThreadID;

			clientInfo.clientAddr		= clientAddr;
			clientInfo.hClientSocket	= hClientSocket;

			std::cout << "Client connected from " << inet_ntoa(clientAddr.sin_addr) << std::endl;

			// Start the client thread
			hClientThread = CreateThread(NULL, 0,
				(LPTHREAD_START_ROUTINE)ClientThread,
				(LPVOID)&clientInfo, 0, &dwThreadID);

			if (hClientThread == NULL)
				std::cout << "Unable to create client thread" << std::endl;
			else
				CloseHandle(hClientThread);
		}
	}

	// Set a signal to close all client connections

	closesocket(hServerSocket);
	WSACleanup();
	return 0;
}

bool InitWinSock2_0()
{
	WSADATA wsaData;
	WORD wVersion = MAKEWORD(2, 0);

	// If WSA Start up is successful, return true, else false
	if (!WSAStartup(wVersion, &wsaData)) return true;
	return false;
}

BOOL WINAPI ClientThread(LPVOID lpData)
{
	CLIENT_INFO* pClientInfo = (CLIENT_INFO*)lpData;
	char buffer[BUFFER_SIZE];
	int nLength = 0;
	int counter = 0;
	int constexpr LIMIT = 8196;	// Just a random number to check for some reason if nLength keep returning -1, once reach this limit return TRUE to get out of this thread

	// check if the user name specified by this current client is already in used
	while (true)
	{
		nLength = recv(pClientInfo->hClientSocket, buffer, sizeof(buffer), 0);
		if (nLength > 0)
		{	// inside this scope, check if the user name entered by user already exists
			// szBuffer here will contain the username from the user
			// if username exist, continue loop and send a message back to user to prompt for a new username
			counter = 0;
			SetNullTerminator(buffer, nLength);
			if (users.find(buffer) != users.end())
			{
				SetMessage(buffer, "[Username has already been used. Please enter another name.]");
				SendMessageToClient(buffer, pClientInfo);
				continue;
			}
		}
		else // nLength is -1 for some reason, just continue until limit
		{
			if (LIMIT <= ++counter)
				return TRUE;	// get out of function
			continue;
		}
		
		/*
			idea is to lock the unordered_map "users" so that no there
			will be no data race when multiple users join at the same time
		*/
		wait();

		std::string const& userName = buffer;
		pClientInfo->userName = userName;

		memset(buffer, 0, BUFFER_SIZE);
		// Send a message to all other connected users welcoming new client
		sprintf(buffer, "[%s joined]", userName.c_str());
		SendMessageToAllClient(buffer);

		// insert new user into the unordered_map
		users[userName] = *pClientInfo;

		signal();
		break;	// break out of while loop
	}

	// Send a message to connected user
	memset(buffer, 0, BUFFER_SIZE);
	sprintf(buffer, "[Welcome %s!]", pClientInfo->userName.c_str());
	SendMessageToClient(buffer, pClientInfo);

	pClientInfo = &users[pClientInfo->userName];

	while (true)
	{
		nLength = recv(pClientInfo->hClientSocket, buffer, sizeof(buffer), 0);
		if (nLength > 0)
		{
			SetNullTerminator(buffer, nLength);	
			std::cout << "Received \"" << buffer << "\" from " << pClientInfo->userName << " (" << inet_ntoa(pClientInfo->clientAddr.sin_addr) << ")" << std::endl;

			if (!strcmp(buffer, "@quit")) 
			{	// the received string buffer is @quit, hence we close the client
				std::cout << pClientInfo->userName << " has terminated their program!" << std::endl;	

				memset(buffer, 0, BUFFER_SIZE);
				sprintf(buffer, "[%s exited]", pClientInfo->userName.c_str());

				SendMessageToAllClient(buffer);

				closesocket(pClientInfo->hClientSocket);
				users.erase(pClientInfo->userName);
				return TRUE;
			}
			else if (!strcmp(buffer, "@names"))
			{
				// send the name of all connected users to the client who requested it
				SetMessage(buffer, "[Connected users: ");
				for (auto const& it : users)
				{
					CLIENT_INFO const& clientInfo = it.second;
					strcat(buffer, clientInfo.userName.c_str());
					strcat(buffer, ", ");
				}
				size_t const LENGTH = strlen(buffer);
				*(buffer + LENGTH - 2) = ']';
				*(buffer + LENGTH - 1) = '\0';
				SendMessageToClient(buffer, pClientInfo);
			}
			else
			{	// No command issued, send message to all connected users
				/*
					1) Append name to the message
					2) Send the message to all connected users
				*/
				std::string const& message = buffer;
				memset(buffer, 0, BUFFER_SIZE);
				sprintf(buffer, "[%s]: %s", pClientInfo->userName.c_str(), message.c_str());
				SendMessageToAllClient(buffer);
			}
		}
		else
			std::cout << "Error reading the data from " << inet_ntoa(pClientInfo->clientAddr.sin_addr) << std::endl;
	}

	return TRUE;
}

void SendMessageToClient(char* pBuffer, CLIENT_INFO const* pClientInfo)
{
	/*
		send() may not be able to send the complete data in one go.
		So try sending the data in multiple requests

		This while loop attempts to send the data in one shot,
		if it doesn't then it will send it multiple times until it succeeds
	*/
	int nCntSend = 0;
	int nLength = strlen(pBuffer);
	while ( (nCntSend = send(pClientInfo->hClientSocket, pBuffer, nLength, 0) != nLength) )
	{
		if (nCntSend == -1)
		{
			std::cout << "Error sending the data to " << inet_ntoa(pClientInfo->clientAddr.sin_addr) << std::endl;
			break;
		}
		if (nCntSend == nLength) break;

		pBuffer += nCntSend;
		nLength -= nCntSend;
	}
}

void SendMessageToAllClient(char* pBuffer)
{
	for (auto const& it : users)
	{
		CLIENT_INFO const& clientInfo = it.second;
		SendMessageToClient(pBuffer, &clientInfo);
	}
}

void SetMessage(char* pBuffer, char const* message)
{
	memset(pBuffer, 0, BUFFER_SIZE);
	strcpy(pBuffer, message);
}

void SetNullTerminator(char* pBuffer, size_t max)
{
	if(max < BUFFER_SIZE)
		*(pBuffer + max) = '\0';
}

void wait(void)
{
	while (s <= 0);
	--s;
}

void signal(void)
{
	++s;
}