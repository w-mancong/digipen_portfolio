#include <WinSock2.h>
#include <iostream>
#include <chrono>
#include <thread>

char szServerIPAddr[] = "127.0.0.1";	// Put here the IP Address of the server
int nServerPort = 5050;

bool InitWinSock2_0();
void ReceieveMessageFromServer(void);
int CreateThreadToReceieveMessage(void);
void SendMessageToServer(char const* pBuffer);
void SetNullTerminator(char* pBuffer, size_t max);
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType);

size_t constexpr BUFFER_SIZE = 1024;
/*
	appStatus: Used to terminate ReceieveMessageFromServer thread when the program ends
	1: Client program is still running
	0: User typed "@quit", prompting to end the program

	keyboardStatus: Used to check if there is any activity of the keyboard (used for pretty printing onto the console on client side
	1: After the cin line
	0: Before the cin line
*/
std::atomic<int> appStatus = 1;
SOCKET hClientSocket;
std::string username{};

int main(void)
{
	if (SetConsoleCtrlHandler(CtrlHandler, TRUE))
	{
		std::cout << "Enter the server IP Address: ";
		std::cin >> szServerIPAddr;
		std::cout << "Enter the server port number: ";
		std::cin >> nServerPort;

		if (!InitWinSock2_0())
		{
			std::cout << "Unable to Initialize Windows Socket environment" << WSAGetLastError() << std::endl;
			return -1;
		}

		hClientSocket = socket(
			AF_INET,			// The address family. AF_INET specifies TCP/IP
			SOCK_STREAM,		// Protocol type. SOCK_STREAM specifies TCP
			0					// Protocol Name. Should be 0 for AF_INET address family
		);

		if (hClientSocket == INVALID_SOCKET)
		{
			std::cout << "Unable to create Server socket" << std::endl;
			// Cleanup the environment initialized by WSAStartup()
			WSACleanup();
			return -1;
		}

		// Create the structure describing various Server parameters
		struct sockaddr_in serverAddr;

		serverAddr.sin_family		= AF_INET;	// The address family. MUST BE AF_INET
		serverAddr.sin_addr.s_addr	= inet_addr(szServerIPAddr);
		serverAddr.sin_port			= htons(nServerPort);

		// Connect client to the server
		if (connect(hClientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
		{	// Failed to connect client to server address
			std::cout << "Unable to connect to " << szServerIPAddr << " on port " << nServerPort << std::endl;
			closesocket(hClientSocket);
			return -1;
		}

		char buffer[BUFFER_SIZE] = {}; // creating a buffer to store the messages from client and send over to server
		// At this point, client have yet to join the chat room
		while (true)
		{	// Will loop till server confirms that there is no duplicate usernames
			//memset(buffer, 0, BUFFER_SIZE);
			std::cout << "Enter username: ";
			std::cin >> username;

			// Send message to server
			SendMessageToServer(username.c_str());

			// After sending message to server, receieve confirmation for user name
			int nLength = 0;
			nLength = recv(hClientSocket, buffer, sizeof(buffer), 0);	// recv returns the length of the received msg, -1 if error receiving message
			SetNullTerminator(buffer, nLength);
			if (nLength > 0)
			{
				/*
					Two types of message will be received from the server
					1) [Username has already been used. Please enter another name.]
					2) [Welcome %s!]
				*/
				std::cout << buffer << std::endl;
				if (buffer[1] == 'U')   continue;
				else if (buffer[1] == 'W') break;
			}
		}

		// Create thread to run the function ReceieveMessageFromServer
		if (CreateThreadToReceieveMessage() == -1)
			return -1;	// failed to create thread

		std::cout << "Enter the string to send (type @quit to exit): ";
		// Infinite loop to send message
		while (true)
		{
			std::cin.getline(buffer, BUFFER_SIZE);	// To receive the user input

			SendMessageToServer(buffer);

			/*
				if user enters @quit, the commend to leave the app then
				break out of the loop to close the client socket
			*/
			if (!strcmp(buffer, "@quit"))
			{	// Since the client application will also be threaded (so that it can receive message even when waiting for input from user)
				appStatus = 0;	// set the variable appStatus to 0 when user type in the command @quit
				break;
			}

			// Let main thread sleep for 5000 microseconds before asking user for input
			std::this_thread::sleep_for(std::chrono::microseconds(5000));
		}
	}

	closesocket(hClientSocket);
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

void ReceieveMessageFromServer(void)
{
	// Thread to receive message from server will always be running as long as appStatus is 1
	char buffer[BUFFER_SIZE] = "";
	// Infinite loop to receive message
	while (appStatus)
	{
		int nLength = 0;
		nLength = recv(hClientSocket, buffer, sizeof(buffer), 0);
		SetNullTerminator(buffer, nLength);
		if (nLength > 0)
		{	
			// This code segment here is just for pretty printing onto the console
			// The user that is receiving this message is also the one who sent this message
			if(std::string(buffer).find(username.c_str()) != std::string::npos)
				std::cout << buffer << std::endl << "Enter the string to send (type @quit to exit): ";
			else
				std::cout << std::endl << buffer << std::endl << "Enter the string to send (type @quit to exit): ";
		}
		memset(buffer, 0, BUFFER_SIZE);
	}
}

int CreateThreadToReceieveMessage(void)
{
	HANDLE hClientThread;
	DWORD dwThreadID;

	hClientThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)ReceieveMessageFromServer, nullptr, 0, &dwThreadID);

	if (hClientThread == NULL)
	{
		std::cout << "Unable to create client thread" << std::endl;
		return -1;
	}
	else
		CloseHandle(hClientThread);
	return 1;
}

void SendMessageToServer(char const* pBuffer)
{
	/*
		send() may not be able to send the complete data in one go.
		So try sending the data in multiple requests
	
		This while loop attempts to send the data in one shot,
		if it doesn't then it will send it multiple times until it succeeds
	*/
	int nCntSend = 0;
	int nLength = strlen(pBuffer);
	while ( ( nCntSend = send( hClientSocket, pBuffer, nLength, 0 ) != nLength ) )
	{
		if (nCntSend == -1)
		{
			std::cout << "Error sending the data to server" << std::endl;
			break;
		}
		if (nCntSend == nLength) break;

		pBuffer += nCntSend;
		nLength -= nCntSend;
	}
}

void SetNullTerminator(char* pBuffer, size_t max)
{
	if (max < BUFFER_SIZE)
		*(pBuffer + max) = '\0';
}

BOOL __stdcall CtrlHandler(DWORD fdwCtrlType)
{
	char buffer[] = "@quit";
	switch (fdwCtrlType)
	{
		// When window console x button is pressed
	case CTRL_CLOSE_EVENT:
		appStatus = 0;
		SendMessageToServer(buffer);
		return TRUE;
	default:
		return FALSE;
	}
}
