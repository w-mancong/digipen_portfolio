/*!*****************************************************************************
\file splitter.cpp
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
#include <fstream>
#include <string>
#include <vector>

constexpr int FOUR_K{4096};

struct Split
{
  int sizeChunks;
  std::string outputFileName;
  std::string inputFileName;
} split;

struct Join
{
  std::string outputFileName;
  std::vector<std::string> inputFileNames;
} join;

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
  void *ptr_struct{nullptr};
  CommandType command_type = parse_commands(&ptr_struct, &rs, argc, argv);

  switch (command_type)
  {
  case CommandType::SPLIT:
  {
    // Split sd = *(Split*)input_struct -> C language
    rs = split_file(&*static_cast<Split *>(ptr_struct));
    break;
  }
  case CommandType::JOIN:
  {
    rs = join_files(&*static_cast<Join *>(ptr_struct));
    break;
  }
  default:
    break;
  }

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
  CommandType command_type{CommandType::NONE};
  size_t const max_count = static_cast<size_t>(argc);
  for (size_t i = 0; i < max_count; ++i)
  {
    std::string argument_value{*(argv + i)};
    if (argument_value == "-s")
    {
      *output_struct = &split;
      split.sizeChunks = std::atoi(*(argv + i + 1));
      if (split.sizeChunks <= 0)
      {
        *rs = SplitResult::E_SMALL_SIZE;
        return CommandType::NONE;
      }

      command_type = CommandType::SPLIT;
      break;
    }
    else if (argument_value == "-j")
    {
      *output_struct = &join;
      command_type = CommandType::JOIN;
      break;
    }
  }
  // check if output_struct is still nullptr
  if (!output_struct)
  {
    *rs = SplitResult::E_NO_ACTION;
    return CommandType::NONE;
  }

  switch (command_type)
  {
  case CommandType::SPLIT:
  {
    // find output file directory and output file name (-o)
    for (size_t i = 0; i < max_count; ++i)
    {
      std::string argument_value{*(argv + i)};
      if (argument_value != "-o")
        continue;
      std::string str{*(argv + i + 1)};
      // the next line is another command prompt
      if (str[0] == '-')
        break;
      split.outputFileName = str;
      break;
    }

    if (split.outputFileName == "")
    {
      *rs = SplitResult::E_NO_ACTION;
      return CommandType::NONE;
    }

    // find input file name (-i)
    for (size_t i = 0; i < max_count; ++i)
    {
      std::string argument_value{*(argv + i)};
      if (argument_value != "-i")
        continue;
      // the next line is another command prompt
      if (*(*(argv + i + 1)) == '-')
        break;
      split.inputFileName = *(argv + i + 1);
      break;
    }

    if (split.inputFileName == "")
    {
      *rs = SplitResult::E_NO_ACTION;
      return CommandType::NONE;
    }

    break;
  }
  case CommandType::JOIN:
  {
    // find output file directory (-o)
    for (size_t i = 0; i < max_count; ++i)
    {
      std::string argument_value{*(argv + i)};
      if (argument_value != "-o")
        continue;
      // next line is another command prompt
      if (*(*(argv + i + 1)) == '-')
        break;
      join.outputFileName = *(argv + i + 1);
      break;
    }

    if (join.outputFileName == "")
    {
      *rs = SplitResult::E_NO_ACTION;
      return CommandType::NONE;
    }

    // find input file names (-i)
    for (size_t i = 0; i < max_count; ++i)
    {
      std::string argument_value{*(argv + i)};
      if (argument_value != "-i")
        continue;
      for (size_t j = i + 1; j < max_count; ++j)
      {
        std::string fileName{*(argv + j)};
        // next line is another command prompt
        if (fileName[0] == '-')
          break;
        join.inputFileNames.push_back(fileName);
      }
      break;
    }

    if (join.inputFileNames[0] == "")
    {
      *rs = SplitResult::E_NO_ACTION;
      return CommandType::NONE;
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
  std::ifstream ifs(sd->inputFileName, std::ifstream::binary);
  if (!ifs)
    return SplitResult::E_BAD_SOURCE;
  int counter{1};
  ifs.seekg(0, ifs.end);
  int totalFileSize = ifs.tellg();
  ifs.seekg(0, ifs.beg);

  int const BUFFER_SIZE = sd->sizeChunks < FOUR_K ? sd->sizeChunks : FOUR_K;

  while (totalFileSize > 0)
  {
    // break out of the loop when the flag is <= 0
    int flag{totalFileSize < sd->sizeChunks ? totalFileSize : sd->sizeChunks};
    // To append the file name with 0001, 0002, etc..
    char outputFileName[2048]{};
    sprintf(outputFileName, "%s%04d", sd->outputFileName.c_str(), counter++);
    std::ofstream ofs(outputFileName, std::ofstream::binary);
    if (!ofs)
      return SplitResult::E_BAD_DESTINATION;

    while (flag > 0)
    {
      // Read from file
      char buffer[FOUR_K]{'\0'};
      int const size = flag >= BUFFER_SIZE ? BUFFER_SIZE : flag;
      ifs.read(buffer, size);

      ofs.write(buffer, size);
      flag -= size;
    }

    // Make size chunks become smaller for the condition to break
    totalFileSize -= sd->sizeChunks;
  }
  return SplitResult::E_SPLIT_SUCCESS;
}

/*!*****************************************************************************
    \brief To join splitted files

    \param [in] jd: Struct containing all the relevant data from command line
                arguments to join files together

    \return Final result of joining files
*******************************************************************************/
SplitResult join_files(Join const *const jd)
{
  std::ofstream ofs(jd->outputFileName, std::ofstream::binary);
  if (!ofs)
    return SplitResult::E_BAD_DESTINATION;
  for (size_t i = 0; i < jd->inputFileNames.size(); ++i)
  {
    std::ifstream ifs(jd->inputFileNames[i], std::ifstream::binary);
    if (!ifs)
      return SplitResult::E_BAD_SOURCE;

    ifs.seekg(0, ifs.end);
    int totalFileSize = ifs.tellg();
    ifs.seekg(0, ifs.beg);

    int const BUFFER_SIZE = totalFileSize < FOUR_K ? totalFileSize : FOUR_K;

    while (totalFileSize > 0)
    {
      // read from file
      char buffer[FOUR_K]{};
      int const size = totalFileSize >= BUFFER_SIZE ? BUFFER_SIZE : totalFileSize;
      ifs.read(buffer, size);

      // write into file
      ofs.write(buffer, size);
      totalFileSize -= size;
    }
  }
  return SplitResult::E_JOIN_SUCCESS;
}
