/*!*****************************************************************************
\file q.hpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 2
\date 21-01-2022
\brief
This programs read data from a text file and transform them into a summary
of information in a table. The functions include:
- largest_max_ht
Helper function to get the maximum height

- avg_max_ht
Helper function to calculate the maximum height

- read_tsunami_data
Read data from a text file and store them into an array of struct

- print_tsunami_data
Print the extracted data from the text file and print them into a table
*******************************************************************************/
#ifndef	Q_HPP
#define Q_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace hlp2
{
	struct Tsunami
	{
		Tsunami(void) : day(0), month(0), year(0), num_fatalities(0), max_wave_height(0.0), location("") {};
		~Tsunami(void) {};

		int day, month, year;
		int num_fatalities;
		double max_wave_height;
		std::string location;
	};
	
	/*!*****************************************************************************
    	\brief
    	Read data of tsunami and store them into an array of struct
    	\param[in] file_name
    	Name of the file to be opened
    	\param[in, out] max_cnt 
    	Size of the array created
	*******************************************************************************/
	Tsunami* read_tsunami_data(std::string const& file_name, int& max_cnt);

	/*!*****************************************************************************
    	\brief
    	From the data extracted from the text file, format it and print it out
		as a table
    	\param[in] arr
    	Pointer to the first element of Tsunami array
    	\param[in] size
    	Total elements in the Tsunami array
		\param[in] file_name
		Name of the file to be outputed
	*******************************************************************************/			
	void	 print_tsunami_data(Tsunami const* arr, int size, std::string const& file_name);
}

#endif