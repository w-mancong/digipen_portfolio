/*!*****************************************************************************
\file splitter.h
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 1
\date 07-09-2022
\brief
This file contain function declarations that parse commands, split files and then
joining them back together
*******************************************************************************/
#ifndef SPLITTER_H
#define SPLITTER_H

#include <stddef.h> 

typedef enum 
{
  E_BAD_SOURCE = 1,
  E_BAD_DESTINATION,
  E_NO_MEMORY,
  E_SMALL_SIZE,
  E_NO_ACTION,
  E_SPLIT_SUCCESS,
  E_JOIN_SUCCESS
} SplitResult;

typedef enum
{
  SPLIT,
  JOIN,
  NONE
} CommandType;

typedef struct Split Split;
typedef struct Join Join;

// provide function header documentation using Doxygen tags to explain to
// your client how they must use this function to either split big file into
// smaller files or join [previously split] smaller files into big file ...
#ifdef __cplusplus
extern "C" 
{
#endif
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
  SplitResult split_join(int argc, char *argv[]);

  /*!*****************************************************************************
      \brief Parsing commands pass by user

      \param [out] output_struct: pointer to store address of struct
      \param [out] rs: store the result of parsing the command
      \param [in] argc: total number of commands
      \param [in] argv: argument values of command

      \return Type of command, Split/Join
  *******************************************************************************/
  CommandType parse_commands(void **output_struct, SplitResult *rs, int argc, char *argv[]);

  /*!*****************************************************************************
      \brief Split files into chunks with size specified by the user

      \param [in] sd: Struct containing all the relevant data from command line
                  arguments to split files apart

      \return Final result of splitting files
  *******************************************************************************/
  SplitResult split_file(Split const *const sd);

  /*!*****************************************************************************
      \brief To join splitted files

      \param [in] jd: Struct containing all the relevant data from command line
                  arguments to join files together

      \return Final result of joining files
  *******************************************************************************/
  SplitResult join_files(Join const *const jd);
#ifdef __cplusplus
}
#endif

#endif // end of #ifndef SPLITTER_H
