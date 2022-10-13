/*!*****************************************************************************
\file uShell.c
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Operating System
\par Assignment 2
\date 14-10-2022
\brief
This file contains function that mimics a shell program
*******************************************************************************/
#define _POSIX_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

#define MAX_BUFFER      1024
#define MAX_ARGUMENTS   40  // max argument of 40, last element should be null terminated
#define MAX_COMMANDS    80  // max command length in an argument
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*array))

typedef struct Variable Variable;
size_t var_size = 10, current_var_index = 0;

typedef struct Process Process;
size_t pro_size = 10, current_pro_index = 0;

char temp_buffer[MAX_BUFFER];
char *args[MAX_ARGUMENTS];
// there is 40 argument values, where each commands can be 80 characters long + 1 null character
char argv[MAX_ARGUMENTS + 1][MAX_COMMANDS + 1];

typedef enum bool
{
    false,
    true,
} bool;

typedef enum Commands
{
    INVALID = -1,
    ECHO,
    EXIT,
    SETVAR,
    FINISH,
    HISTORICAL,
    EXTERNAL,
} Commands;

typedef enum EchoState
{
    NO_STATE = -1,
    SUCCEED,    // Managed to find a key and is returning the value associated with it
    NOT_A_KEY,  // Searching through to realised that it's not a key
    FAILURE,    // No such key exist yet
} EchoState;

struct Variable
{
    char *key;
    char *value;
} *var = NULL;

struct Process
{
    size_t index;
    pid_t pid;
    bool child_process;
} *process = NULL;

/*!*****************************************************************************
    \brief Prints out msg and flush stdout stream

    \param [in] msg: Message to be printed out onto the screen
*******************************************************************************/
void Print(char const *msg);

/*!*****************************************************************************
    \brief Read in characters from the standard input file

    \param [out] str: Inputs from standard input file will be stored in this 
    char array
    \param [in] max: Maximum characters to be read from the input stream
*******************************************************************************/
void ReadIn(char str[], size_t max);

/*!*****************************************************************************
    \brief To determine the different internal commands that uShell supports

    \param [in] buffer: String buffer of input read from input stream
    \param [out] next_index: To store the index position after reading in the
    internal command

    \return The enum type of command that was parse
*******************************************************************************/
Commands Parse(char const *buffer, size_t *next_index);

/*!*****************************************************************************
    \brief Echo a message by user onto the shell

    \param [in] buffer: String buffer containing the message to be filtered 
    and echo onto the terminal
*******************************************************************************/
void Echo(char const *buffer);

/*!*****************************************************************************
    \brief To filter any special variable and have it's message replaced and
    echoed onto the terminal

    \param [in] buffer: String buffer containing the message to be filtered
    \param [out] str: Store the filtered message to be echoed into this char 
    array
*******************************************************************************/
void EchoMessage(char const *buffer, char str[]);

/*!*****************************************************************************
    \brief To increase the size storing all the special variables
*******************************************************************************/
void ResizeVariable(void);

/*!*****************************************************************************
    \brief To add/change special variables that contain another message

    \param [in] buffer: String buffer containing data of key and value of the
    special variable
*******************************************************************************/
void SetVariable(char const *buffer);

/*!*****************************************************************************
    \brief Search to see if buffer contains a key that's within my var container
    holding a value to it

    \param [in] buffer: String buffer containing a potential key
    \param [out] state: Echo state after searching the string buffer and var 
    container. The job of each state are:
    SUCCEED,    // Managed to find a key and is returning the value associated with it
    NOT_A_KEY,  // Searching through to realised that it's not a key
    FAILURE,    // No such key exist yet
    \param [out] len: Length of the key string

    \return The value string that the key string is associated with the special
    variable, else NULL will be returned
*******************************************************************************/
char *SearchKeyValue(char const *buffer, EchoState *state, size_t *len);

/*!*****************************************************************************
    \brief Terminates the process

    \param [in] buffer: String buffer containing the process index to be 
    terminated
*******************************************************************************/
void Finish(char const *buffer);

/*!*****************************************************************************
    \brief Release all memory allocated on the heap for var and process
*******************************************************************************/
void FreeMemory(void);

/*!*****************************************************************************
    \brief General function that handles external commands

    \param [in] buffer: String buffer containing external commands
*******************************************************************************/
void External(char const* buffer);

/*!*****************************************************************************
    \brief Does pipe communication between two different processes

    \param [in] buffer: String buffer containing two different programs where
    they will communicate between each other using pipe
    \param [in] pipe_index: Index position of |
*******************************************************************************/
void Pipe(char const *buffer, size_t pipe_index);

/*!*****************************************************************************
    \brief To increase the size storing all the processes
*******************************************************************************/
void ResizeProcess(void);

int main(void)
{
    bool should_run = true;
    // Default for variable container
    var = (Variable *)malloc(sizeof(Variable) * var_size);
    memset(var, 0, sizeof(Variable) * var_size);

    // Default for process container
    process = (Process *)malloc(sizeof(Process) * pro_size);
    memset(process, 0, sizeof(Process) * pro_size);

    // Default for temp buffer
    memset(temp_buffer, 0, sizeof(temp_buffer));

    // For historical program
    char historical[MAX_BUFFER];
    memset(historical, 0, sizeof(historical));
    bool run_last_command = false;

    Print("\033[1;31mWelcome to Man Cong's Shell Program!\033[0m\n");

    while (should_run)
    {
        /*
            https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal 
            -> Change output text of linux terminal
        */
        Print("\033[0;32muShell>\033[0;33m");
        char buffer[MAX_BUFFER] = {'\0'};
        if(!run_last_command)
            ReadIn(buffer, MAX_BUFFER);
        else
        {
            strcpy(buffer, historical);
            strcat(historical, "\n");
            Print(historical);
        }

        run_last_command = false;

        size_t next_index = 0;
        Commands command = Parse(buffer, &next_index);
        switch (command)
        {
        case ECHO:
        {
            Echo(buffer + next_index);
            break;
        }
        case EXIT:
        {
            should_run = false;
            break;
        }
        case SETVAR:
        {
            SetVariable(buffer + next_index);
            break;
        }
        case FINISH:
        {
            Finish(buffer + next_index);
            break;
        }
        case HISTORICAL:
        {
            if(strcmp(historical, ""))
                run_last_command = true;
            // No history commands
            else
                Print("Error: No commands in history buffer.\n");
            break;
        }
        case EXTERNAL:
        {
            memset(args, 0, sizeof(args));
            memset(argv, 0, sizeof(argv));
            External(buffer);
            break;
        }
        default:
            break;
        }

        // Copy latest command into the historical buffer
        if(!run_last_command && strcmp(buffer, "!!"))
            strcpy(historical, buffer);
    }

    FreeMemory();
    return 0;
}

/***********************************************************
                Internal Functionalities
***********************************************************/
void Print(char const *msg)
{
    printf("%s", msg);
    fflush(stdout);
}

// Own fgets so can truncate everything at the back
void ReadIn(char str[], size_t max)
{
    char ch = '\0';
    size_t index = 0;
    while ((ch = getchar()) != '\n' && ch != EOF)
    {
        if (index >= max - 1)
            break;
        *(str + index++) = ch;
    }
    *(str + index) = '\0';
}

/***********************************************************
                        PARSE
***********************************************************/
Commands Parse(char const *buffer, size_t *next_index)
{
    // commands are case sensitive
    char const *COMMANDS[] = 
    {
        "echo",
        "exit",
        "setvar",
        "finish",
        "!!",
    };

    // remove any white spaces in front
    char const *ptr = buffer;
    while (isspace(*ptr))
        ++ptr;

    char read_command[MAX_BUFFER];
    size_t index = 0;
    // Extracting the command from buffer
    while (true)
    {
        if (isspace(*ptr) || *ptr == '\0')
            break;
        *(read_command + index++) = *ptr++;
    }
    // Adding a null terminator at the end of the string
    *(read_command + index++) = '\0';

    size_t const TOTAL_COMMANDS = ARRAY_SIZE(COMMANDS);
    // Differentiating the different internal commands
    for (size_t i = 0; i < TOTAL_COMMANDS; ++i)
    {
        // check if any of the read command belongs to any internal commands
        if (strcmp(read_command, *(COMMANDS + i)))
            continue;
        *next_index = index;
        return (Commands)(i);
    }

    *next_index = 0;
    // Might be used for external commands
    return EXTERNAL;
}

/***********************************************************
                            ECHO
***********************************************************/
void Echo(char const *buffer)
{
    char temp[MAX_BUFFER];
    memset(temp, 0, sizeof(temp));
    EchoMessage(buffer, temp);
    Print(temp);
}

void EchoMessage(char const *buffer, char str[])
{
    size_t i = 0;
    EchoState state = NO_STATE;
    const char *ptr = buffer;
    /*
        Initial removal any white spaces in front
        echo           ok
            ^^^^^^^^^^^
        All these white spaces will be removed
    */
    while (isspace(*ptr))
        ++ptr;
    while (*ptr)
    {
        // Possibility of it being a key that needs to be replaced
        if (*ptr == '$' && *(ptr + 1) == '{')
        {
            size_t len = 0;
            char *val = SearchKeyValue(ptr + 2, &state, &len);
            switch (state)
            {
            case FAILURE:
            {
                sprintf(str, "Error: %s is not a defined variable\n", val);
                memset(temp_buffer, 0, sizeof(temp_buffer));
                return;
            }
            case SUCCEED:
            {
                strcat(str, val); // append replacement value to str
                i = strlen(str);  // next index for str is the len of str
                ptr += len;       // let pointer point to the next ones
                continue;
            }
            default:
                break;
            }
        }

        /*
            Second removal of white spaces from the echo command
            echo   okasd       abced
                ^^^     ^^^^^^
            First three white spaces will be removed by the first white space removal
            This while loop here is to remove all additional white spaces but only leaving one white space
        */
        while (isspace(*ptr) && isspace(*(ptr + 1))) ++ptr;

        *(str + i++) = *ptr++; // Assigning the echo message into the str buffer
    }
    *(str + i++) = '\n', *(str + i) = '\0'; // putting a newline and null terminating character at the end of str
}

/***********************************************************
                        SETVAR
***********************************************************/
void ResizeVariable(void)
{
    // increase the size by two times
    var_size <<= 1;
    // Create a temp with more space allocated
    Variable *temp = (Variable *)malloc(sizeof(Variable) * var_size);
    memset(temp, 0, sizeof(Variable) * var_size);
    size_t const OLD_SIZE = var_size >> 1;
    // swap values in var into temp
    for (size_t i = 0; i < OLD_SIZE; ++i)
    {
        (temp + i)->key = (var + i)->key;
        (temp + i)->value = (var + i)->value;
        // setting var's key and value to nullptr
        (var + i)->key = NULL;
        (var + i)->value = NULL;
    }
    free(var);
    var = temp;
}

void SetVariable(char const *buffer)
{
    /*
        uShell>setvar HAHA hoohoo # assign the value hoohoo to HAHA
                      ^
                      |
                     ptr is pointing here
    */
    const char *ptr = buffer;
    // remove any white spaces
    while (isspace(*ptr)) ++ptr;

    char key[MAX_BUFFER] = {'$', '{'}, value[MAX_BUFFER] = {'\0'}; // use to store the key and value respectively
    size_t i = 2, j = 0, k = 0;
    // Extract key data
    while (!isspace(*ptr) && *ptr != '\0') // iterate everything then when it's a space/null character break out
        *(key + i++) = *ptr++;
    // assign last char to be a null character
    *(key + i++) = '}', *(key + i++) = '\0';

    if(strlen(key) <= 3)
        Print("Error: Variable name cannot be empty\n");

    // Extra value data
    ++ptr;
    while (true)
    {
        if (*ptr == '\0')
            break;
        *(value + j++) = *ptr++;
    }
    *(value + j) = '\0';

    Variable *vptr = NULL;
    // Search to check if there vars has any key with the same key here, if have then change value instead of adding new one
    while (k < current_var_index)
    {
        if (strcmp(key, (var + k++)->key))
            continue;
        vptr = var + --k;
        break;
    }

    // if key is found (change the value)
    if (vptr)
    {
        size_t const VAL_SIZE = sizeof(char) * (strlen(value) + 1);

        // Allocate appropriate memory for new value
        char *temp = (char *)malloc(VAL_SIZE);
        // Copy new value into temp first
        strcpy(temp, value);

        // deallocate old value memory and assign it with temp
        free(vptr->value);
        vptr->value = temp;

        temp = NULL;
    }
    // no such key is found (add into the container)
    else
    {
        // Resize the "vector" if vector no longer has any storage
        if (current_var_index >= var_size)
            ResizeVariable();

        size_t const KEY_SIZE = sizeof(char) * (strlen(key) + 1);
        size_t const VAL_SIZE = sizeof(char) * (strlen(value) + 1);

        // Allocate appropriate memory for key and value
        (var + current_var_index)->key = (char *)malloc(KEY_SIZE);
        (var + current_var_index)->value = (char *)malloc(VAL_SIZE);
        strcpy((var + current_var_index)->key, key);

        if (1 < VAL_SIZE)
            strcpy((var + current_var_index)->value, value);
        // Give default value if value is empty
        else
            *(var + current_var_index)->value = '\0';

        ++current_var_index;
    }
}

char *SearchKeyValue(char const *buffer, EchoState *state, size_t *len)
{
    // Search the entire buffer to see if there is a closing bracer, if dh means it's no a key
    const char *ptr = buffer;
    do
    {
        if(*ptr == '}')
            break;
    } while (*++ptr);
    
    // ptr reached the end of buffer and didn't find a closing bracer, therefore this buffer doesn't contain a key
    if(!*ptr)
    {
        *state = NOT_A_KEY;
        return NULL;
    }

    // using this to store the extracted key value from buffer
    char key[MAX_BUFFER] = { '$', '{' };
    size_t index = 2; size_t const KEY_LEN = ptr - buffer;
    for (size_t i = 0; i <= KEY_LEN; ++i)
    {
        /*
            uShell>echo ${HAHA }123 # wrong use of curly braces, Would be read as 2 words.
            ${HAHA }123
        */
        if(isspace(*(buffer + i)) || *(buffer + i) == '$')
        {
            *state = NOT_A_KEY;
            return NULL;
        }
        *(key + index++) = *(buffer + i);
    }

    // After extracting key from buffer, search key to see if key exists
    for (size_t i = 0; i < current_var_index; ++i)
    {
        if(!strcmp((var + i)->key, key))
        {
            *len = strlen((var + i)->key);
            *state = SUCCEED;
            return (var + i)->value;
        }
    }

    // No such key exists, so len should just be 0
    *len = 0;
    *state = FAILURE;
    // Getting the name of the key
    for (size_t i = 2, j = 0; i < KEY_LEN + 2; ++i, ++j)
        *(temp_buffer + j) = *(key + i);

    // return the name of the key so that error message can be printed
    return temp_buffer;
}

/***********************************************************
                            Finish
***********************************************************/
void Finish(char const *buffer)
{
    /*
        uShell>finish 1
        process 10064 exited with exit status 0.
    */
    char const *ptr = buffer;

    size_t i = 0;
    while(*ptr && !isspace(*ptr))
        *(temp_buffer + i++) = *ptr++;

    size_t process_index = (size_t)atoll(temp_buffer) - 1;
    // user input contains a process_index that exists
    if(current_pro_index > process_index)
    {
        memset(temp_buffer, 0, sizeof(temp_buffer));
        if ((process + process_index)->child_process)
        {
            if(kill( (process + process_index)->pid, SIGKILL ) == -1)
            {
                sprintf(temp_buffer, "Error: %s\n", strerror(errno));
                Print(temp_buffer);
            } 
            else
            {
                int status;
                waitpid((process + process_index)->pid, &status, 0);
                if(WIFEXITED(status))
                {
                    // no longer a child process
                    (process + process_index)->child_process = false;
                    sprintf(temp_buffer, "Process %d exited with exit status %d.\n", (process + process_index)->pid, WEXITSTATUS(status));
                    Print(temp_buffer);
                }
            }
        }
        else
        {
            sprintf(temp_buffer, "Process Index %ld with Process ID %d is no longer a child process\n", (process + process_index)->index, (process + process_index)->pid);
            Print(temp_buffer);
        }
    }
    // process_index does not exists
    else
        Print("Error: No such process index exist\n");
    memset(temp_buffer, 0, sizeof(temp_buffer));
}

/***********************************************************
                Free Dynamic Allocated Memory
***********************************************************/
void FreeMemory(void)
{
    for (size_t i = 0; i < var_size; ++i)
    {
        if ((var + i)->key)
            free((var + i)->key);
        if ((var + i)->value)
            free((var + i)->value);
    }
    if (var)
        free(var);
    if(process)
        free(process);
}

/***********************************************************
                External Functionalities
***********************************************************/
void External(char const* buffer)
{
    // Search for | to establish pipe
    char const *ptr = buffer;
    bool pipe_communication = false;
    size_t pipe_index = 0;
    do
    {
        if(*(ptr + 1) == ' ' && *(ptr - 1) == ' ' && *ptr == '|')
        {
            pipe_communication = true;
            pipe_index = ptr - buffer;
            break;
        }
    } while (*++ptr);

    if(pipe_communication)
    {
        Pipe(buffer, pipe_index);
        return;
    }

    /*
        uShell>cat prog.c &
               ^
               |
              ptr pointing here
    */
    bool parent_wait = true;
    size_t ampersand_position = ULLONG_MAX;
    // search for isolated &
    ptr = buffer;
    do
    {
        if (*ptr != '&')
            continue;
        if (*(ptr - 1) != ' ' || (*(ptr + 1) != ' ' && *(ptr + 1) != '\0'))
            continue;
        parent_wait = false;
        ampersand_position = ptr - buffer;
        break;
    } while (*++ptr);

    // search for any redirection
    size_t redirect_input = 0, redirect_output = 0;
    bool redirection = false;
    char inputFile[MAX_BUFFER >> 1], outputFile[MAX_BUFFER >> 1];
    memset(inputFile , 0, sizeof(inputFile));
    memset(outputFile, 0, sizeof(outputFile));
    /*
        uShell>sort < in.txt > out.txt.
               ^
               |
              ptr pointing here
    */
    ptr = buffer;
    do
    {
        if(*(ptr + 1) == ' ' && *(ptr - 1) == ' ')
        {
            if(*ptr == '<')
            {
                redirect_input  = ptr - buffer;
                redirection = true;
            }
            else if(*ptr == '>')
            {
                redirect_output = ptr - buffer;
                redirection = true;
            }
        }
    } while (*++ptr);

    // Extract program name and arguments from buffer
    size_t i = 0, j = 0;
    ptr = buffer;
    // There is no redirection needed
    if(!redirection || ampersand_position < redirect_input || ampersand_position < redirect_output)
    {
        bool increment = true;
        while (*ptr)
        {
            if (MAX_ARGUMENTS < i)
            {
                Print("Error: Too many command arguments\n");
                return;
            }

            /*
                Removal of any white spaces
                uShell>        cat       prog.c
                       ^^^^^^^^   ^^^^^^^
                These white spaces will be removed

                uShell>cat prog.c & > input.txt // this case is wrong lol
            */
            while (isspace(*ptr)) ++ptr;
            if (*ptr == '\0') break;

            increment = true;
            // Extract relevant data
            for (j = 0; j < MAX_COMMANDS + 1; ++j)
            {
                if (isspace(*ptr) || *ptr == '\0')
                    break;
                if (*ptr == '&' && !parent_wait)
                {
                    ++ptr;
                    increment = false;
                    continue;
                }
                *(*(argv + i) + j) = *ptr++;
            }

            // Error handling to make sure the size of command is should
            if (MAX_COMMANDS < j)
            {
                Print("Error: Length of command is too long\n");
                return;
            }

            // Place a null terminator at the end of the string
            *(*(argv + i) + j) = '\0';
            *(args + i) = *(argv + i);
            if (increment)
                ++i;
        }

        *(args + i) = NULL;
    }
    // Command to be parse have the ampersand at the very end
    else
    {
        bool to_iterate = true;
        while(to_iterate)
        {
            if(MAX_ARGUMENTS < i)
            {
                Print("Error: Too many command arguments\n");
                return;
            }

            /*
                Removal of any white spaces
                uShell>        cat       prog.c > out.txt &
                       ^^^^^^^^   ^^^^^^^
                These white spaces will be removed
            */
            while(isspace(*ptr)) ++ptr;            

            // Extract relevant data
            for (j = 0; j < MAX_COMMANDS + 1; ++j)
            {
                if(isspace(*ptr))
                    break;
                if (*ptr == '<' || *ptr == '>')
                {
                    to_iterate = false;
                    break;
                }
                *(*(argv + i) + j) = *ptr++;
            }

            // Error handling to make sure the size of command is should
            if (MAX_COMMANDS < j)
            {
                Print("Error: Length of command is too long\n");
                return;
            }

            if(!to_iterate)
                continue;
            // Place a null terminator at the end of the string
            *(*(argv + i) + j) = '\0';
            *(args + i) = *(argv + i);
            ++i;
        }
        *(args + i) = NULL;

        /*
            Extract input file and output file location
            uShell>sort < in.txt
                        ^
                        |
                  redirect_input index

            uShell>sort < in.txt > out.txt
                        ^        ^
                        |        |
                  redirect_input index
                                 |
                      redirect_output index

            Only accepted format to have child process run without parent process waiting
            uShell>cat prog.c > out.txt &
        */
        if(0 < redirect_input)
        {
            /*
                uShell>sort < in.txt > out.txt
                              ^
                              |
                             ptr is pointing here
            */
            size_t k = 0;
            ptr = buffer + redirect_input + 2;
            /*
                Removal of any whitespace
                uShell>sort <             in.txt > out.txt
                              ^^^^^^^^^^^^ These white spaces will be removed
                              |
                             ptr is pointing here
            */
            while(isspace(*ptr)) ++ptr;
            // Extracting the input file name and storing it in inFile
            while(*ptr)
            {
                if(isspace(*ptr)) break;
                *(inputFile + k++) = *ptr++;
            }
        }
        if(0 < redirect_output)
        {
            /*
            uShell>sort < in.txt > out.txt
                                   ^
                                   |
                                  ptr is pointing here
            */
            size_t k = 0;
            ptr = buffer + redirect_output + 2;
            /*
                Removal of any whitespace
                uShell>sort < in.txt >          out.txt
                                      ^^^^^^^^^^^ These white spaces will be removed
                                                |
                                               ptr is pointing here
            */
            while (isspace(*ptr)) ++ptr;
            while(*ptr)
            {
                if(isspace(*ptr)) break;
                *(outputFile + k++) = *ptr++;
            }
        }
    }

    int fd[2];
    if(pipe(fd) == -1)
    {
        Print("Error: Pipe failed\n");
        return;
    }

    pid_t pid = fork();
    int error = 0;
    // Creation of a process failed
    if(pid == -1)
    {
        Print("Error: Failed to create a child process\n");
        return;
    }
    // Child process
    else if(pid == 0)
    {
        close(fd[0]);
        if (redirection)
        {
            int inFile = 0, outFile = 0;
            if(0 < redirect_input)
            {
                inFile = open(inputFile, O_RDONLY, 0777);
                if(inFile == -1)
                {
                    sprintf(temp_buffer, "Error: %s\n", strerror(errno));
                    Print(temp_buffer);
                    memset(temp_buffer, 0, sizeof(temp_buffer));
                    exit(1);
                }
                error = dup2(inFile, STDIN_FILENO);
                if(error == -1)
                {
                    sprintf(temp_buffer, "Error: %s\n", strerror(errno));
                    Print(temp_buffer);
                    memset(temp_buffer, 0, sizeof(temp_buffer));
                    exit(1);
                } 
                close(inFile);
            }
            if(0 < redirect_output)
            {
                outFile = open(outputFile, O_WRONLY | O_CREAT | O_TRUNC, 0777);
                if(outFile == -1)
                {
                    sprintf(temp_buffer, "Error: %s\n", strerror(errno));
                    Print(temp_buffer);
                    memset(temp_buffer, 0, sizeof(temp_buffer));
                    exit(1);
                }
                error = dup2(outFile, STDOUT_FILENO);
                if (error == -1)
                {
                    sprintf(temp_buffer, "Error: %s\n", strerror(errno));
                    Print(temp_buffer);
                    memset(temp_buffer, 0, sizeof(temp_buffer));
                    exit(1);
                }
                close(outFile);
            }
        }

        error = execvp(*args, args);
        if(write(fd[1], &error, sizeof(int)) == -1)
        {
            Print("Error: Unable to write to pipe\n");
            exit(1);
        }
        close(fd[1]);
        // If unable to run the new program, reset everything then exit from child process
        if(error == -1)
        {
            sprintf(temp_buffer, "Error: %s cannot be found\n", *args);
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(1);
        }
    }
    // Parent process
    else if(pid > 0) 
    {
        close(fd[1]);
        if(read(fd[0], &error, sizeof(int)) == -1)
        {
            Print("Error: Unable to read from pipe\n");
            return;
        }
        close(fd[0]);

        if(parent_wait)
        {
            int status = 0;
            waitpid(pid, &status, 0);
        }
        // if dont have to wait for child process, add process and relevant stuff
        else
        {            
            if(error == -1)
                return;

            if (current_pro_index >= pro_size)
                ResizeProcess();

            (process + current_pro_index)->index = current_pro_index + 1;
            (process + current_pro_index)->pid = pid;
            (process + current_pro_index)->child_process = true;

            sprintf(temp_buffer, "[%ld] %d\n", (process + current_pro_index)->index, (process + current_pro_index)->pid);
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));

            ++current_pro_index;
        }
    }
}

void Pipe(char const *buffer, size_t pipe_index)
{
    char const *ptr = buffer;
    size_t i = 0, j = 0;
    pid_t left_child = 0, right_child = 0;

    int fd[2];  // 0 -> read, 1 -> write
    if(pipe(fd) == -1)
    {
        Print("Error: Pipe failed\n");
        return;
    }

    /*
      Extract program name of left hand side first
      uShell>ls -l | less    //  left_child process | right_child process
    */
    while (*ptr)
    {
        if ((size_t)(ptr - buffer) + 1 >= pipe_index)
            break;

        /*
            uShell>            ls -l | less
                   ^^^^^^^^^^^^ Removing these white spaces
        */
        while (isspace(*ptr)) ++ptr;

        // Extract relevant data
        for (j = 0; j < MAX_COMMANDS + 1; ++j)
        {
            if (isspace(*ptr))
                break;
            *(*(argv + i) + j) = *ptr++;
        }
        // Place a null terminator at the end of the string
        *(*(argv + i) + j) = '\0';
        *(args + i) = *(argv + i);
        ++i;
    }
    *(args + j) = NULL;

    int error = 0;
    // Forking to run program on the left
    left_child = fork();
    if(left_child == -1)
    {
        Print("Error: Failed to create a left child process\n");
        return;
    }    
    // Child process
    if(left_child == 0)
    {
        close(fd[0]);   // close read becuz left child doesn't need to read anything
        error = dup2(fd[1], STDOUT_FILENO);
        if (error == -1)
        {
            sprintf(temp_buffer, "Error: %s\n", strerror(errno));
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(1);
        }
        close(fd[1]);

        // Execute program
        error = execvp(*args, args);
        // If unable to run the new program, reset everything then exit from child process
        if (error == -1)
        {
            sprintf(temp_buffer, "Error: %s cannot be found\n", *args);
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(1);
        }
    }

    int status = 0;
    waitpid(left_child, &status, 0);

    // child process did not exit normally
    if(WIFEXITED(status) == 0)
    {
        Print("Error: Left child process did not terminate normally\n");
        return;
    }

    // Reset variables for use to extract program for right_child
    memset(args, 0, sizeof(args));
    memset(argv, 0, sizeof(argv));
    j = 0, i = 0;
    ptr = buffer + pipe_index + 1;
    while (*ptr)
    {
        /*
            uShell>ls -l |      less
                          ^^^^^^ Removing these white spaces
        */
        while (isspace(*ptr)) ++ptr;

        // Extract relevant data
        for (j = 0; j < MAX_COMMANDS + 1; ++j)
        {
            if (isspace(*ptr))
                break;
            *(*(argv + i) + j) = *ptr++;
        }
        // Place a null terminator at the end of the string
        *(*(argv + i) + j) = '\0';
        *(args + i) = *(argv + i);
        ++i;
    }
    *(args + j) = NULL;

    // fork right child to do the process
    right_child = fork();
    if (right_child == -1)
    {
        Print("Error: Failed to create a left child process\n");
        return;
    }
    // Child process
    if (right_child == 0)
    {
        close(fd[1]); // close read becuz left child doesn't need to read anything
        error = dup2(fd[0], STDIN_FILENO);
        if (error == -1)
        {
            sprintf(temp_buffer, "Error: %s\n", strerror(errno));
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(1);
        }
        close(fd[0]);

        // Execute program
        error = execvp(*args, args);
        // If unable to run the new program, reset everything then exit from child process
        if (error == -1)
        {
            sprintf(temp_buffer, "Error: %s cannot be found\n", *args);
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(1);
        }
    }

    close(fd[0]);
    close(fd[1]);
    waitpid(right_child, &status, 0);

    // child process did not exit normally
    if (WIFEXITED(status) == 0)
    {
        Print("Error: Left child process did not terminate normally\n");
        return;
    }
}

void ResizeProcess(void)
{
    // Store the old size
    size_t const OLD_SIZE = pro_size;
    // Increase size of container by 2 times
    pro_size <<= 1;
    // Allocate a temporary memory
    Process *temp = (Process *)malloc(sizeof(Process) * pro_size);
    // Give temp storage a default value
    memset(temp, 0, sizeof(Process) * pro_size);
    // Move the data from old process to new temporary container
    memmove(temp, process, sizeof(Process) * OLD_SIZE);
    // Deallocate old container
    free(process);
    // Let process point to the new continer
    process = temp;
    // Temp should not be pointing to any memory after this point
    temp = NULL;
}
