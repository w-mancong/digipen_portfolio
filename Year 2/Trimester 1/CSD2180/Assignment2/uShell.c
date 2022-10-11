#define _POSIX_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <setjmp.h>

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

void Print(char const *msg);
void ReadIn(char str[], size_t max);
Commands Parse(char const *buffer, size_t *next_index);
void Echo(char const *buffer);
void EchoMessage(char const *buffer, char str[]);
void ResizeVariable(void);
void SetVariable(char const *buffer);
char *SearchKeyValue(char const *buffer, EchoState *state, size_t *len);
void Finish(char const *buffer);
void FreeMemory(void);
void External(char const* buffer);
void ResizeProcess(void);

int main(void)
{
    bool should_run = true;
    var = (Variable *)malloc(sizeof(Variable) * var_size);
    memset(var, 0, sizeof(Variable) * var_size);

    process = (Process *)malloc(sizeof(Process) * pro_size);
    memset(process, 0, sizeof(Process) * pro_size);

    memset(temp_buffer, 0, sizeof(temp_buffer));

    char historical[MAX_BUFFER];
    memset(historical, 0, sizeof(historical));
    bool run_last_command = false;

    // struct sigaction sa;
    // void delete_zombies(void);

    // sigfillset(&sa.sa_mask);
    // sa.sa_handler = delete_zombies;
    // sa.sa_flags = 0;
    // sigaction(SIGCHLD, &sa, NULL);

    Print("Welcome to Man Cong's Shell Program!\n");

    while (should_run)
    {
        Print("uShell>");
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

        /* read a command from the keyboard */
        /* parse the command */

        /*
         * After parsing, the steps are:
         * For internal comments:
         * (1) invoke corresponding functions
         * For external comments:
         * (1) fork a child process using fork()
         * (2) the child process will invoke execve()
         * (3) parent will invoke wait() unless command included &
         */

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
    char const *COMMANDS[] = {
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
        while (isspace(*ptr) && isspace(*(ptr + 1)))
            ++ptr;

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
    /*
        uShell>cat prog.c
    */
    char const *ptr = buffer, *sptr = buffer;
    bool parent_wait = true, increment = true;

    // search for isolated &
    do
    {
        if (*sptr != '&')
            continue;
        if (*(sptr - 1) != ' ' || (*(sptr + 1) != ' ' && *(sptr + 1) != '\0'))
            continue;
        parent_wait = false;
        break;
    } while (*++sptr);
    
    // Extract program name and arguments from buffer
    size_t i = 0, j = 0;
    while(*ptr)
    {
        if(MAX_ARGUMENTS < i)
        {
            Print("Error: Too many command arguments\n");
            return;
        }
        
        /*
            Removal of any white spaces
            uShell>        cat       prog.c
                   ^^^^^^^^   ^^^^^^^
            These white spaces will be removed
        */
        while(isspace(*ptr)) ++ptr;

        increment = true;
        // Extract relevant data
        for (j = 0; j < MAX_COMMANDS + 1; ++j)
        {
            if(isspace(*ptr) || *ptr == '\0')
                break;
            if(*ptr == '&')
            {
                ++ptr;
                increment = false;
                continue;
            }
            *(*(argv + i) + j) = *ptr++;
        }

        // Error handling to make sure the size of command is should
        if(MAX_COMMANDS < j)
        {
            Print("Error: Length of command is too long\n");
            return;
        }

        // Place a null terminator at the end of the string
        *(*(argv + i) + j) = '\0';
        *(args + i) = *(argv + i);
        if(increment)
            ++i;
    }

    *(args + i) = NULL;
    int fd[2];
    if(pipe(fd) == -1)
    {
        Print("Error: Pipe failed\n");
        return;
    }

    pid_t pid = fork();
    // Creation of a process failed
    if(pid <= -1)
    {
        Print("Error: Failed to create a child process\n");
        return;
    }
    // Child process
    else if(pid == 0)
    {
        close(fd[0]);
        int error = execvp(*args, args);
        if(write(fd[1], &error, sizeof(int)) == -1)
        {
            Print("Error: Unable to write to pipe\n");
            return;
        }
        close(fd[1]);
        // If unable to run the new program, reset everything then exit from child process
        if(error == -1)
        {
            sprintf(temp_buffer, "Error: %s cannot be found\n", *args);
            Print(temp_buffer);
            memset(temp_buffer, 0, sizeof(temp_buffer));
            exit(0);
        }
    }
    // Parent process
    else if(pid > 0) 
    {
        close(fd[1]);
        int error = 0;
        if(read(fd[0], &error, sizeof(int)) == -1)
        {
            Print("Error: Unable to write from pipe\n");
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
