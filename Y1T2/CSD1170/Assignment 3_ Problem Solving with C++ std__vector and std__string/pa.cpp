/*!*****************************************************************************
\file pa.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Assignment 3
\date 27-01-2022
\brief 
This program reads a text file and store the data of cities and population
into a vector of structs, then sorting them based on accending and decending
order of their names and population. After sorting, the program will output
the result into a text file.
*******************************************************************************/
#include "pa.hpp"

namespace Helper
{
	/**************************************************************************/
  	/*!
    \brief
    	Remove specified character from string
    
    \param [in, out] str
		String to have all of char c removed in it's array
    
    \param [in] c
		Character to be removed from the string
  	*/
  	/**************************************************************************/
	void Remove(std::string& str, char const& c)
	{
		for (std::string::iterator it = str.begin(); it != str.end(); ++it)
		{
			if (*it == c)
				str.erase(it);
		}
	}
}

namespace HLP2
{
	/**************************************************************************/
  	/*!
    \brief
		Load and store data into a vector of CountryInfo
    
    \param [in] is
      	Input stream to receive the data from
    
    \return
		A vector filled with CountryInfo initialised with data retrieved
		from is
  	*/
  	/**************************************************************************/
	std::vector<CountryInfo> fill_vector_from_istream(std::istream& is)
	{
		std::string buffer;
		std::vector<CountryInfo> countryInfos;
		const char* upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", *lower = "abcdefghijklmnopqrstuvwxyz()", *number = "1234567890";
		while (std::getline(is, buffer))
		{
			std::string num = buffer.substr(buffer.find_first_of(number), buffer.find_last_of(number) - buffer.find_first_of(number) + 1);
			Helper::Remove(num, ',');
			countryInfos.push_back(CountryInfo(buffer.substr(buffer.find_first_of(upper), buffer.find_last_of(lower) - buffer.find_first_of(upper) + 1), std::stoll(num.c_str())));
		}

		return countryInfos;
	}

  	/**************************************************************************/
  	/*!
    \brief
		Return the length of the longest country's name
    
    \param [in] countryInfos
		A vector of CountryInfo with data of all the countries and population

	\return
		Longest country's name
  	*/
  	/**************************************************************************/
	size_t max_name_length(std::vector<CountryInfo> const& countryInfos)
	{
		size_t max_len = 0;
		for (std::vector<CountryInfo>::const_iterator it = countryInfos.begin(); it != countryInfos.end(); ++it)
		{
			if (it->name.size() > max_len)
				max_len = it->name.size();
		}
		return max_len;
	}

	/**************************************************************************/
  	/*!
    \brief
      	Selection sort based on pointer function passed in
    
    \param [in] rv
      	Reference to the vector of CountryInfo to be sorted
    
    \param [in] cmp
      	Pointer to the specific function to have it's comparison executed
  	*/
  	/**************************************************************************/
	void sort(std::vector<CountryInfo>& rv, Ptr_Cmp_Func cmp)
	{
		for (size_t i = 0; i < rv.size() - 1; ++i)
		{
			size_t index = i;
			for (size_t j = i + 1; j < rv.size(); ++j)
			{
				if (cmp(rv[j], rv[i]))
				{
					size_t temp = index;
					index = j;
					if (cmp(rv[temp], rv[index]))
						index = temp;
				}
			}
			std::swap(rv[i], rv[index]);
		}
	}

  	/**************************************************************************/
  	/*!
    \brief
    	Write to a specific output stream
    
    \param [in] v
		Reference to a vector of CountryInfo to have it's sorted data written
		to an output stream
    
    \param [in, out] os
		Reference to the output stream to be written into
    
    \param [in] fw
		Width to be offsetted
  	*/
  	/**************************************************************************/
	void write_to_ostream(std::vector<CountryInfo> const& v, std::ostream& os, size_t fw)
	{
		os << std::left << std::setfill(' ');
		for (std::vector<CountryInfo>::const_iterator it = v.begin(); it != v.end(); ++it)
			os << std::setw(fw) << it->name << it->pop << std::endl;
	}

	/**************************************************************************/
  	/*!
    \brief
		Compare the name of the two countries. 
    
    \param [in] left
		Left hand side index
    
    \param [in] right
		Right hand side index
    
    \return
		Returns true if left is lexicographically less than right
  	*/
  	/**************************************************************************/
	bool cmp_name_less(CountryInfo const& left, CountryInfo const& right)
	{
		const int SHORTEST_LEN = left.name.size() < right.name.size() ? left.name.size() : right.name.size();
		for (int i = 0; i < SHORTEST_LEN; ++i)
		{
			if (left.name[i] < right.name[i])
				return true;
			if (left.name[i] != right.name[i])
				return false;
		}
		if(left.name.size() < right.name.size())
			return true;
		return false;
	}

	/**************************************************************************/
  	/*!
    \brief
		Compare the name of the two countries. 
    
    \param [in] left
		Left hand side index
    
    \param [in] right
		Right hand side index
    
    \return
		Returns true if left is lexicographically more than right
  	*/
  	/**************************************************************************/
	bool cmp_name_greater(CountryInfo const& left, CountryInfo const& right)
	{
		const int SHORTEST_LEN = left.name.size() < right.name.size() ? left.name.size() : right.name.size();
		for (int i = 0; i < SHORTEST_LEN; ++i)
		{
			if (left.name[i] > right.name[i])
				return true;
			if (left.name[i] != right.name[i])
				break;
		}
		return false;
	}

	/**************************************************************************/
  	/*!
    \brief
		Compare the population of the two countries. 
    
    \param [in] left
		Left hand side index
    
    \param [in] right
		Right hand side index
    
    \return
		Returns true if left is numerically less than right
  	*/
  	/**************************************************************************/
	bool cmp_pop_less(CountryInfo const& left, CountryInfo const& right)
	{
		return left.pop < right.pop;
	}

	/**************************************************************************/
  	/*!
    \brief
		Compare the population of the two countries. 
    
    \param [in] left
		Left hand side index
    
    \param [in] right
		Right hand side index
    
    \return
		Returns true if left is numerically more than right
  	*/
  	/**************************************************************************/
	bool cmp_pop_greater(CountryInfo const& left, CountryInfo const& right)
	{
		return left.pop > right.pop;
	}
}