#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h>
#include <string.h>
#include <ctype.h>

#define MAX_BUFFER 1024
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*array))

typedef struct Variable Variable;
size_t var_size = 10, current_var_index = 0;

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
} * var;

void Print(char const *msg);
void ReadIn(char str[], size_t max);
Commands Parse(char const *buffer, size_t *next_index);
void Echo(char const *buffer, size_t index);
void VariablesDefault(Variable **ptr, size_t size);
void ResizeVariable(void);
void SetVariable(char const *buffer, size_t index);
void EchoMessage(char const* buffer, char str[]);
char *SearchKeyValue(char const *buffer, EchoState *state, size_t *len);
void FreeMemory(void);

int main(void)
{
    bool should_run = true;
    var = (Variable *)(malloc(sizeof(Variable) * var_size));
    VariablesDefault(&var, var_size);

    while (should_run)
    {
        Print("uShell>");
        char buffer[MAX_BUFFER] = {'\0'};
        ReadIn(buffer, MAX_BUFFER);

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
            Echo(buffer, next_index);
            break;
        }
        case EXIT:
        {
            should_run = false;
            break;
        }
        case SETVAR:
        {
            SetVariable(buffer, next_index);
            break;
        }
        default:
            break;
        }
    }

    FreeMemory();
    return 0;
}

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
    return INVALID;
}

/***********************************************************
                            ECHO
***********************************************************/
void Echo(char const *buffer, size_t index)
{
    char temp[MAX_BUFFER];
    EchoMessage(buffer + index, temp);
    Print(temp);
}

/***********************************************************
                        SETVAR
***********************************************************/
void VariablesDefault(Variable **ptr, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        (*ptr + i)->key = NULL;
        (*ptr + i)->value = NULL;
    }
}

void ResizeVariable(void)
{
    // increase the size by two times
    var_size <<= 1;
    // Create a temp with more space allocated
    Variable *temp = (Variable *)malloc(sizeof(Variable) * var_size);
    VariablesDefault(&temp, var_size);
    size_t const OLD_SIZE = var_size >> 1;
    // letting new containing point to key and value
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

void SetVariable(char const *buffer, size_t index)
{
    /*
        uShell>setvar HAHA hoohoo # assign the value hoohoo to HAHA
                      ^
                      |
                     ptr is pointing here
    */
    const char *ptr = buffer + index;
    // remove any white spaces
    while (isspace(*ptr)) ++ptr;

    char key[MAX_BUFFER] = {'$', '{'}, value[MAX_BUFFER] = {'\0'}; // use to store the key and value respectively
    size_t i = 2, j = 0, k = 0;
    // Extract key data
    while (!isspace(*ptr) && *ptr != '\0') // iterate everything then when it's a space/null character break out
        *(key + i++) = *ptr++;
    // assign last char to be a null character
    *(key + i++) = '}', *(key + i++) = '\0';
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

void EchoMessage(char const *buffer, char str[])
{
    /*
        uShell>echo ${HAHA} # calling out the value of HAHA
        hoohoo
        uShell>echo ${Haha}123 # Attempting to call out the value of an undefined variable.
        Error: Haha123 is not a defined variable.
        uShell>echo ${${HAHA}} # nested use of curly braces are not supported
        ${hoohoo}
        uShell>echo $${HAHA} # $ sign can be used together with variables
        $hoohoo
    */
    size_t i = 0, len = 0;
    EchoState state = NO_STATE;
    const char *ptr = buffer;
    while(*ptr)
    {
        if(*ptr == '$' && *(ptr + 1) == '{')
        {
            char* val = SearchKeyValue(ptr + 2, &state, len);
            switch(state)
            {
                case FAILURE:
                {
                    sprintf(str, "Error: %s is not a defined variable\n", val);
                    return;
                    break;
             
                }
                case SUCCEED:
                {
                    strcat(str, val);
                    i = strlen(str);
                    ptr += len;     // let pointer point to the next ones
                    continue;
                }
            }
        }
        *(str + i++) = *ptr++;
    }
}

char *SearchKeyValue(char const *buffer, EchoState *state, size_t len)
{
    *state = NO_STATE; // Default value
    // using this to store the key value
    char key[MAX_BUFFER] = { '\0' };

}

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
}
