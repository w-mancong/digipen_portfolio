/*!*****************************************************************************
\file q.cpp
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
#include "q.hpp"

/*!*****************************************************************************
	\brief
	Find the maximum height of a tsunami
	\param[in, out] max_ht 
	Variable to store the max height of tsunami from the data
	\param[in] ht
	Height to be compared with. If it's larger than max_ht, max_ht will be 
	assigned to the value of ht
*******************************************************************************/
void largest_max_ht(double& max_ht, const double& ht)
{
	if (ht > max_ht)
		max_ht = ht;
}

/*!*****************************************************************************
    \brief
    Calculate the average maximum height of all the tsunami
    \param[in, out] avg_ht 
    Variable to store the average height of all the tsunami from the data
    \param[in]	ht
    Height of individual tsunami data
	\param[in] average
	When it reaches the end of the loop, average will be set to true so that
	the average height can be calculated
*******************************************************************************/
void avg_max_ht(double& avg_ht, const double& ht, const bool& average = false)
{
	static int count = 0;
	avg_ht += ht, ++count;
	if(average)
		avg_ht /= count;
}

namespace hlp2
{
/*!*****************************************************************************
    \brief
    Read data of tsunami and store them into an array of struct
    \param[in] file_name
    Name of the file to be opened
    \param[in, out] max_cnt 
    Size of the array created
*******************************************************************************/
	Tsunami* read_tsunami_data(std::string const& file_name, int& max_cnt)
	{
		std::ifstream ifs(file_name);
		if (!ifs.is_open())
		{
			return nullptr;
		}
		std::string buffer;
		max_cnt = 0;
		while (std::getline(ifs, buffer))
		{		
			++max_cnt;
		}
		Tsunami* res = new Tsunami[max_cnt];
		ifs.clear(); ifs.seekg(ifs.beg);
		for(int index = 0; index < max_cnt; ++index)
		{
			ifs >> (*(res + index)).month >> (*(res + index)).day >> (*(res + index)).year >> (*(res + index)).num_fatalities >> (*(res + index)).max_wave_height;

			std::getline(ifs, buffer);
			const char* str = buffer.c_str();
			// Remove whitespaces in front
			while (*str == ' ') ++str;
			if (index < max_cnt)
				(*(res + index)).location = str;
			// Remove whitespace at the back
			for (int i = (*(res + index)).location.size() - 1; i >= 0; --i)
			{
				if ((*(res + index)).location[i] == ' ')
					continue;
				(*(res + index)).location.erase(i + 1);
				break;
			}
		}
		return res;
	}

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
	void print_tsunami_data(Tsunami const* arr, int size, std::string const& file_name)
	{	
		std::ofstream ofs(file_name);
		ofs << "List of tsunamis:\n" << "-----------------\n";
		ofs << std::right << std::fixed << std::setprecision(2) ;
		double max_ht = 0.0, avg_ht = 0.0;
		for (int i = 0; i < size; ++i)
		{
			ofs << std::setw(2)  << std::setfill('0') << (*(arr + i)).month << ' ' << std::setw(2) << (*(arr + i)).day << ' ' << (*(arr + i)).year;
			ofs << std::setw(7)  << std::setfill(' ') << (*(arr + i)).num_fatalities;
			ofs << std::setw(11) << (*(arr + i)).max_wave_height;
			ofs << "     " << (*(arr + i)).location << "\n";

			largest_max_ht(max_ht, (*(arr + i)).max_wave_height);
			avg_max_ht(avg_ht, (*(arr + i)).max_wave_height, size - 1 == i ? true : false);
		}
		ofs << "\nSummary information for tsunamis\n--------------------------------\n\n";
		ofs << "Maximum wave height (in meters): " << max_ht << "\n\n";
		ofs << "Average wave height (in meters): " << std::setw(5) << avg_ht << "\n\n";
		ofs << "Tsunamis with greater than average height " << avg_ht << ":\n";
		for (int i = 0; i < size; ++i)
		{
			if ((*(arr + i)).max_wave_height > avg_ht)
				ofs << (*(arr + i)).max_wave_height << "     " << (*(arr + i)).location << std::endl;
		}
	}
}