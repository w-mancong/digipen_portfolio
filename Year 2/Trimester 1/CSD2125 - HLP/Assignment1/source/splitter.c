/*!*****************************************************************************
\file splitter.c
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 1
\date 07-09-2022
\brief
This file contain function definition to parse commands, split files and then
joining them back together
*******************************************************************************/
#include "splitter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FOUR_K 4096 

typedef struct Split
{
  int sizeChunks;
  char *outputFileName;
  char *inputFileName;
} Split;

typedef struct Join
{
  char *outputFileName;
  char **inputFileNames;
  size_t totalFiles;
} Join;

Split split;
Join join;

/*!*****************************************************************************
    \brief Free any memory that was allocated by the heap
*******************************************************************************/
void FreeMemory()
{
  if (split.inputFileName)
    free(split.inputFileName);
  if (split.outputFileName)
    free(split.outputFileName);

  if (join.outputFileName)
    free(join.outputFileName);
  for (size_t i = 0; i < join.totalFiles; ++i)
  {
    if (*(join.inputFileNames + i))
      free(*(join.inputFileNames + i));
  }
  if (join.inputFileNames)
    free(join.inputFileNames);
}

/*!*****************************************************************************
    \brief Function to be called from main to split/join files based on commands

    \param [in] argc: total argument counts
    \param [in] argv: string arguments from command prompt

    \return 1. E_NO_ACTION : If user input is not sufficient [number of command-line parameters is not
               sufficient or -j  switch is missing or -s  switch is missing].
            2. E_BAD_SOURCE : If input file for either split or join doesn't exist.
            3. E_BAD_DESTINATION : If output file for either split or join cannot be created.
            4. E_SMALL_SIZE : If byte size of split files is negative or zero.
            5. E_NO_MEMORY : If malloc  returns NULL  [only in splitter.c ].
            6. E_SPLIT_SUCCESS : If a file has been successfully split into smaller chunks.
            7. E_JOIN_SUCCESS : If chunks have been successfully merged into a single file
*******************************************************************************/
SplitResult split_join(int argc, char *argv[])
{
  SplitResult rs = E_NO_ACTION;
  void *ptr_struct = NULL;
  CommandType command_type = parse_commands(&ptr_struct, &rs, argc, argv);

  switch (command_type)
  {
  case SPLIT:
  {
    rs = split_file(&*(Split *)ptr_struct);
    break;
  }
  case JOIN:
  {
    rs = join_files(&*(Join *)ptr_struct);
    break;
  }
  default:
    break;
  }
  FreeMemory();
  return rs;
}

/*!*****************************************************************************
    \brief Parsing commands pass by user

    \param [out] output_struct: pointer to store address of struct
    \param [out] rs: store the result of parsing the command
    \param [in] argc: total number of commands
    \param [in] argv: argument values of command

    \return Type of command, Split/Join
*******************************************************************************/
CommandType parse_commands(void **output_struct, SplitResult *rs, int argc, char *argv[])
{
  // Loop through argv to find -s and -j
  CommandType command_type = NONE;
  size_t const max_count = (size_t)(argc);
  for (size_t i = 0; i < max_count; ++i)
  {
    if (!strcmp(*(argv + i), "-s"))
    {
      *output_struct = &split;
      split.sizeChunks = atoi(*(argv + i + 1));
      if (split.sizeChunks <= 0)
      {
        *rs = E_SMALL_SIZE;
        return NONE;
      }

      command_type = SPLIT;
      break;
    }
    else if (!strcmp(*(argv + i), "-j"))
    {
      *output_struct = &join;
      join.totalFiles = argc - 5;
      command_type = JOIN;
      break;
    }
  }
  // check if output_struct is still NULL
  if (!output_struct)
  {
    *rs = E_NO_ACTION;
    return NONE;
  }

  switch (command_type)
  {
  case SPLIT:
  {
    // find output file directory and output file name (-o)
    for (size_t i = 0; i < max_count; ++i)
    {
      if (strcmp(*(argv + i), "-o"))
        continue;
      // the next line is another command prompt
      if (*(*argv + i + 1) == '-')
        break;
      size_t const len = strlen(*(argv + i + 1));
      split.outputFileName = (char *)malloc(sizeof(char) * len + 1);

      if (!split.outputFileName)
      {
        *rs = E_NO_MEMORY;
        return NONE;
      }

      strcpy(split.outputFileName, *(argv + i + 1));
      *(split.outputFileName + len) = '\0';
      break;
    }

    if (!strcmp(split.outputFileName, ""))
    {
      *rs = E_NO_ACTION;
      return NONE;
    }

    // find input file name (-i)
    for (size_t i = 0; i < max_count; ++i)
    {
      if (strcmp(*(argv + i), "-i"))
        continue;
      // the next line is another command prompt
      if (*(*(argv + i + 1)) == '-')
        break;
      size_t const len = strlen(*(argv + i + 1));
      split.inputFileName = (char *)malloc(sizeof(char) * len + 1);

      if (!split.inputFileName)
      {
        *rs = E_NO_MEMORY;
        return NONE;
      }

      strcpy(split.inputFileName, *(argv + i + 1));
      *(split.inputFileName + len) = '\0';
      break;
    }

    if (!strcmp(split.inputFileName, ""))
    {
      *rs = E_NO_ACTION;
      return NONE;
    }
    break;
  }
  case JOIN:
  {
    // find output file directory (-o)
    for (size_t i = 0; i < max_count; ++i)
    {
      if (strcmp(*(argv + i), "-o"))
        continue;
      // next line is another command prompt
      if (*(*argv + i + 1) == '-')
        break;
      size_t const len = strlen(*(argv + i + 1));
      join.outputFileName = (char *)malloc(sizeof(char) * len + 1);

      if (!join.outputFileName)
      {
        *rs = E_NO_MEMORY;
        return NONE;
      }

      strcpy(join.outputFileName, *(argv + i + 1));
      *(join.outputFileName + len) = '\0';
    }

    if (!strcmp(join.outputFileName, ""))
    {
      *rs = E_NO_ACTION;
      return NONE;
    }

    join.inputFileNames = (char **)malloc(sizeof(char *) * join.totalFiles);
    if (!join.inputFileNames)
    {
      *rs = E_NO_MEMORY;
      return NONE;
    }

    int index = 0;

    // find input file names (-i)
    for (size_t i = 0; i < max_count; ++i)
    {
      if (strcmp(*(argv + i), "-i"))
        continue;
      for (size_t j = i + 1; j < max_count; ++j, ++index)
      {
        // next line is another command prompt
        // if(*(*argv + j) == '-')
        //   break;
        size_t const len = strlen(*(argv + j));
        *(join.inputFileNames + index) = (char *)malloc(sizeof(char) * len + 1);

        if (!*(join.inputFileNames + index))
        {
          *rs = E_NO_MEMORY;
          return NONE;
        }

        strcpy(*(join.inputFileNames + index), *(argv + j));
        *(*(join.inputFileNames + index) + len) = '\0';
      }
      break;
    }

    if (!strcmp(*(join.inputFileNames), ""))
    {
      *rs = E_NO_ACTION;
      return NONE;
    }
    break;
  }
  default:
    break;
  }
  return command_type;
}

/*!*****************************************************************************
    \brief Split files into chunks with size specified by the user

    \param [in] sd: Struct containing all the relevant data from command line
                arguments to split files apart

    \return Final result of splitting files
*******************************************************************************/
SplitResult split_file(Split const *const sd)
{
  FILE *rptr = fopen(sd->inputFileName, "rb");
  if (!rptr)
    return E_BAD_SOURCE;
  int counter = 1;
  fseek(rptr, 0L, SEEK_END);
  int totalFileSize = ftell(rptr);
  rewind(rptr);

  int const BUFFER_SIZE = sd->sizeChunks < FOUR_K ? sd->sizeChunks : FOUR_K;
  char *buffer = (char *)malloc(sizeof(char) * BUFFER_SIZE);

  if (!buffer)
    return E_NO_MEMORY;

  while (totalFileSize > 0)
  {
    // break out of the loop when the flag is <= 0
    int flag = totalFileSize < sd->sizeChunks ? totalFileSize : sd->sizeChunks;
    // To append the file name with 0001, 0002, etc..
    char outputFileName[2048];
    sprintf(outputFileName, "%s%04d", sd->outputFileName, counter++);
    FILE *wptr = fopen(outputFileName, "wb");
    if (!wptr)
      return E_BAD_DESTINATION;

    while (flag > 0)
    {
      // Read from file
      int const size = flag >= BUFFER_SIZE ? BUFFER_SIZE : flag;
      fread(buffer, 1, size, rptr);

      fwrite(buffer, 1, size, wptr);
      flag -= size;
    }

    // Make sure chunks become smaller for the condition to break
    totalFileSize -= sd->sizeChunks;
  }
  free(buffer);
  return E_SPLIT_SUCCESS;
}

/*!*****************************************************************************
    \brief To join splitted files

    \param [in] jd: Struct containing all the relevant data from command line
                arguments to join files together

    \return Final result of joining files
*******************************************************************************/
SplitResult join_files(Join const *const jd)
{
  FILE *wptr = fopen(jd->outputFileName, "wb");
  if (!wptr)
    return E_BAD_DESTINATION;
  for (size_t i = 0; i < jd->totalFiles; ++i)
  {
    FILE *rptr = fopen(*(jd->inputFileNames + i), "rb");
    if (!rptr)
      return E_BAD_SOURCE;

    fseek(rptr, 0L, SEEK_END);
    int totalFileSize = ftell(rptr);
    rewind(rptr);

    while (totalFileSize > 0)
    {
      // read from file
      char buffer[FOUR_K];
      size_t size = fread(buffer, 1, FOUR_K, rptr);

      // write into file
      fwrite(buffer, 1, size, wptr);
      totalFileSize -= size;
    }
  }
  return E_JOIN_SUCCESS;
}
