/*!*****************************************************************************
\file q.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 1
\date 15-01-2022
\brief
This program reads a text file specified by it's parameter input_filename
and prints some statistical results about it's contents to an output file
specified by parameter analysis_file. The functions include:
- seperators
prints out a set amount of separators between each row of data
- q
Opens a file and output statistical results of it's content
*******************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>

namespace hlp2
{
    /*!*****************************************************************************
    \brief
    Separate rows of data by a fixed number of character
    \param[in] c
    Character used as a seperator
    \param[in] ofs
    File stream of the .txt file
    *******************************************************************************/
    void seperators(const char c, std::ofstream& ofs)
    {
        const int TOTAL_SEPARATOR = 70;
        for (int i = 1; i < TOTAL_SEPARATOR; ++i)
            ofs << c;
        ofs << '\n';
    }

    /*!*****************************************************************************
    \brief
    Opens a file and output it's contents as a statistical result
    \param[in] input_filename
    File to be opened
    \param[in] analysis_file
    File that will have the statistical result written into 
    *******************************************************************************/
    void q(char const* input_filename, char const* analysis_file)
    {
        std::ifstream ifs(input_filename);
        if (!ifs.is_open())
        {
            std::cout << "File " << input_filename << " not found." << std::endl;
            ifs.close();
            return;
        }
        std::ofstream ofs(analysis_file);
        if (!ofs.is_open())
        {
            std::cout << "Unable to create output file " << analysis_file << std::endl;
            ofs.close();
            return;
        }

        const int TOTAL_ALPHABETS = 26;
        int lower[TOTAL_ALPHABETS] = { 0 };                   // Stores lower letter
        int upper[TOTAL_ALPHABETS] = { 0 };                   // Stores upper letter
        int white_space = -1, others = 0, digits = 0, total_int = 0, sum_of_int = 0;
        char buffer[1024];

        while (!ifs.eof())
        {        
            ifs.getline(buffer, sizeof(buffer));
            for (int i = 0; *(buffer + i) != '\0'; ++i)
            {
                if (islower(*(buffer + i)))
                    ++lower[*(buffer + i) - 'a'];
                else if (isupper(*(buffer + i)))
                    ++upper[*(buffer + i) - 'A'];
                else if (isspace(*(buffer + i)))
                    ++white_space;
                else if (isdigit(*(buffer + i)))
                {
                    char int_buffer[64] = { '\0' };
                    int j = 0;
                    while (isdigit(*(buffer + i)))
                    {
                        *(int_buffer + j) = *(buffer + i);
                        ++digits; ++i; ++j;
                    }
                    *(int_buffer + j) = '\0';
                    sum_of_int += atoi(int_buffer);
                    --i; ++total_int;
                }
                else
                    ++others;
            }
            ++white_space;
        }

        int total_letters = 0, total_lower = 0, total_upper = 0;
        for (int i = 0; i < TOTAL_ALPHABETS; ++i)
        {
            total_letters += *(lower + i) + *(upper + i);
            total_lower += *(lower + i);
            total_upper += *(upper + i);
        }

        float avg_sum = 0;
        if (total_int)
            avg_sum = sum_of_int / (float)total_int;

        ofs << "Statistics for file: " << input_filename << "\n";
        seperators('-', ofs);
        int sum_category = total_letters + white_space + digits + others;
        ofs << '\n' << "Total # of characters in file: " << sum_category << "\n\nCategory            How many in file             % of file\n";
        seperators('-', ofs);
        ofs << "Letters" << std::setfill(' ') << std::setw(29) << total_letters << std::setw(20) << std::fixed << std::setprecision(2) << (total_letters / (float)sum_category) * 100.0f << " %\n";
        ofs << "White space" << std::setfill(' ') << std::setw(25) << white_space << std::setw(20) << (white_space / (float)sum_category) * 100.0f << " %\n";
        ofs << "Digits" << std::setfill(' ') << std::setw(30) << digits << std::setw(20) << (digits / (float)sum_category) * 100.0f << " %\n";
        ofs << "Other characters" << std::setfill(' ') << std::setw(20) << others << std::setw(20) << (others / (float)sum_category) * 100.0f << " %\n\n\n";
        ofs << "LETTER STATISTICS\n\n";
        ofs << "Category            How many in file      % of all letters\n";
        seperators('-', ofs);
        ofs << "Uppercase" << std::setfill(' ') << std::setw(27) << total_upper << std::setw(20) << (total_upper / (float)total_letters) * 100.0f << " %\n";
        ofs << "Lowercase" << std::setfill(' ') << std::setw(27) << total_lower << std::setw(20) << (total_lower / (float)total_letters) * 100.0f << " %\n";
        char alpabet = 'a';
        for (int i = 0; i < TOTAL_ALPHABETS; ++i)
        {
            int total_alphabet = *(lower + i) + *(upper + i);
            ofs << (char)(alpabet + i) << std::setfill(' ') << std::setw(35) << total_alphabet << std::setw(20) << (total_alphabet / (float)total_letters) * 100.0f << " %\n";
        }
        ofs << "\n\nNUMBER ANALYSIS\n\n";
        ofs << "Number of integers in file:          " << total_int << '\n';
        ofs << "Sum of integers:                     " << sum_of_int << '\n';
        ofs << "Average of integers:                 " << avg_sum << '\n';
        seperators('_', ofs);

        ifs.close();
        ofs.close();
    }
}