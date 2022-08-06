/*!*****************************************************************************
\file pa.hpp
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
#ifndef PA_HPP_
#define PA_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace HLP2
{
	struct CountryInfo
	{
		CountryInfo(std::string name = "", long int pop = 0) : name(name), pop(pop) {};
		~CountryInfo(void) {};
		std::string name;		// Country's name
		long int pop;			// Country's population
	};

	// Same as using typedef
	// Corresponding typedef declaration looks like:
	// typedef bool (*Ptr_Cmp_Func)(CountryInfo const&, CountryInfo const&);
	using Ptr_Cmp_Func = bool (*)(CountryInfo const&, CountryInfo const&);

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
	std::vector<CountryInfo> fill_vector_from_istream(std::istream& is);

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
	size_t max_name_length(std::vector<CountryInfo> const& countryInfos);

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
	void sort(std::vector<CountryInfo>& rv, Ptr_Cmp_Func cmp);

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
	void write_to_ostream(std::vector<CountryInfo> const& v, std::ostream& os, size_t fw);

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
	bool cmp_name_less(CountryInfo const& left, CountryInfo const& right);

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
	bool cmp_name_greater(CountryInfo const& left, CountryInfo const& right);

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
	bool cmp_pop_less(CountryInfo const& left, CountryInfo const& right);

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
	bool cmp_pop_greater(CountryInfo const& left, CountryInfo const& right);
}

#endif