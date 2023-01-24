/*!*****************************************************************************
\file pstree.c
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: Operating System
\par Assignment 1
\date 18-09-2022
\brief
This file contains functions to print out the process tree
*******************************************************************************/
#define _DEFAULT_SOURCE
#define _BSD_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <dirent.h>
#include <pwd.h>
#include <sys/types.h>

typedef struct ps_node ps_node;

struct ps_node
{
    pid_t pid, ppid;
    ps_node *child[256], *next;
    char procName[512];
};

ps_node *head = NULL;

/*!*****************************************************************************
    \brief Remove any white spaces from the string

    \param [out] str: the string to have white spaces removed from it
*******************************************************************************/
void remove_whitespaces(char *str)
{
    char *ptr = str;
    int len = strlen(str);

    // remove any whitespaces at the back
    while (isspace(*(ptr + len - 1)))
        *(ptr + --len) = '\0';
    // iterate the entire string to remove any whitespaces in between
    while (*ptr && isspace(*ptr))
        --len, ++ptr;

    // move the memory from ptr to str
    memmove(str, ptr, len + 1);
}

/*!*****************************************************************************
    \brief Creates a node based on the directoryName

    \param [in] directoryName: Name of the directory
*******************************************************************************/
void create_node(char *directoryName)
{
    char procName[512], fileName[512], buffer[512];
    char *data = NULL, *key = NULL;
    int pid = 0, ppid = 0;
    strcpy(fileName, directoryName);
    strcat(fileName, "/status");
    FILE *file = fopen(fileName, "r");
    if (!file)
        return;

    // retrieving data from process
    while (fgets(buffer, sizeof(buffer), file))
    {
        key = strtok(buffer, ":");
        data = strtok(NULL, ":");
        if (!data || !key)
            continue;
        remove_whitespaces(data); remove_whitespaces(key);
        if (!strcmp(key, "Name"))
            strcpy(procName, data);
        else if (!strcmp(key, "Pid"))
            pid = atoi(data);
        else if (!strcmp(key, "PPid"))
            ppid = atoi(data);
    }

    // creates node
    ps_node *node = (ps_node *)malloc(sizeof(ps_node));
    node->pid = pid;
    node->ppid = ppid;
    size_t child_pointers_size = sizeof(node->child) / sizeof(ps_node *);
    for (size_t i = 0; i < child_pointers_size; ++i)
        *(node->child + i) = NULL;
    node->next = NULL;
    strcpy(node->procName, procName);

    // "Sort" nodes based on their pid by accending order
    if(!head)
    {
        head = node;
        return;
    }

    ps_node *curr = head, *prev = NULL;
    // check if we are supposed to insert in between/front
    while(curr)
    {
        if(node->pid < curr->pid)
        {
            // inserting the the front of the list
            if(!prev)
            {
                head = node;
                head->next = curr;
            }
            // inserting the node between two other
            else
            {
                prev->next = node;
                node->next = curr;
            }
            return;
        }
        prev = curr;
        curr = curr->next;
    }
    // end of the loop
    prev->next = node;
}

/*!*****************************************************************************
    \brief Recursively prints out the process tree

    \param [in] curr: Current node to be printed out
    \param [in] level: Level of which the current node is at
*******************************************************************************/
void print(ps_node *curr, int level)
{
    size_t index = 0;
    while(index < (size_t)level)
        printf(" "), ++index;
    printf("%s [pid: %d, ppid: %d]\n", curr->procName, curr->pid, curr->ppid);
    index = 0;
    while ( *(curr->child + index) )
        print(*(curr->child + index++), level + 1);
}

/*!*****************************************************************************
    \brief Attaching all the children nodes to the parent

    \param [out] curr: Attaching all the children to this node
*******************************************************************************/
void attach_children(ps_node **curr)
{
    ps_node *node = head; size_t index = 0;
    while(node)
    {
        if(node->ppid == (*curr)->pid)
            *((*curr)->child + index++) = node;
        node = node->next;
    }
}

/*!*****************************************************************************
    \brief Recursively loop thru tree to free memory allocated on the heap

    \param [in] n: node to have it's memory freed
*******************************************************************************/
void free_memory(ps_node** n)
{
    if(!(*n))
        return;
    free_memory(&(*n)->next);
    free(*n);
    *n = NULL;
}

/*!*****************************************************************************
    \brief Entrance point of program
*******************************************************************************/
int main(void)
{
    char const *file = "/proc";

    DIR *directoryPtr = opendir(file);
    struct dirent *directoryEntrance = NULL;

    // unable to open the directory
    if (!directoryPtr)
        return 0;

    // read and create nodes
    while ( (directoryEntrance = readdir(directoryPtr)) )
    {
        if (!directoryEntrance || directoryEntrance->d_type != DT_DIR)
            continue;
        char directoryName[512];
        strcpy(directoryName, file); strcat(directoryName, "/"); strcat(directoryName, directoryEntrance->d_name);
        create_node(directoryName);
    }
    // Closing directory because I'm not using it anymore
    closedir(directoryPtr);

    // find children node
    {
        ps_node *curr = head;
        while(curr)
        {
            attach_children(&curr);
            curr = curr->next;
        }
    }

    // print tree
    print(head, 0);

    free_memory(&head);

    return 0;
}
