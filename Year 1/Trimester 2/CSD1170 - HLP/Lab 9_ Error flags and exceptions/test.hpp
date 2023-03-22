/*!*****************************************************************************
\file test.hpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 9
\date 18-03-2022
\brief
This file contains classes that handles division_by_zero and if any of the
input from user is invalid. 

It also contains 5 function definition for us to handle exception cases when
division of two numbers involves the denominator to be 0
*******************************************************************************/
#include <limits>
#include <cerrno>
#include <sstream>
#include <iostream>
#include <exception>

namespace hlp2
{
    /*!*********************************************************************************
    \brief
        invalid_input class inherited from std::exception that will be thrown when 
        istream fails
    ***********************************************************************************/
    class invalid_input : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Returns error message about invalid input from the user when istream fails
        \return
            Error message telling user that there was a invalid input in istream
        *******************************************************************************/
        const char *what(void) const noexcept;
    };

    /*!*****************************************************************************
    \brief
        Returns error message about invalid input from the user when istream fails
    \return
        Error message telling user that there was a invalid input in istream
    *******************************************************************************/
    const char *invalid_input::what() const noexcept
    {
        return "Invalid input!";
    }

    /*!*********************************************************************************
    \brief
        stream_wrapper class inherited from std::exception to handle inputs from user        
    ***********************************************************************************/
    class stream_wrapper : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Constructor for stream_wrapper class
        \param [in] is
            Input stream reference
        *******************************************************************************/
        stream_wrapper(std::istream& is);

        /*!*****************************************************************************
        \brief
            Overloaded right shift operator to handle input from user
        \param [in] rhs
            Value to store into from is
        *******************************************************************************/
        template <typename T>
        stream_wrapper& operator>>(T &rhs);
    private:
        std::istream& is;
    };

    /*!*****************************************************************************
    \brief
        Constructor for stream_wrapper class
    \param [in] is
        Input stream reference
    *******************************************************************************/
    stream_wrapper::stream_wrapper(std::istream &is) : is{ is } {}

    /*!*****************************************************************************
    \brief
        Overloaded right shift operator to handle input from user
    \param [in] rhs
        Value to store into from is
    *******************************************************************************/
    template <typename T>
    stream_wrapper& stream_wrapper::operator>>(T &rhs)
    {
        is >> rhs;
        if(is.fail())
            throw invalid_input();
        return *this;
    }

    /*!*********************************************************************************
    \brief
        division_by_zero inherited from std::exception that will be thrown when 
        denominator is zero
    ***********************************************************************************/
    class division_by_zero : public std::exception
    {
    public:
        /*!*****************************************************************************
        \brief
            Constructor for division_by_zero
        \param [in] numerator
            numerator that is trying to divide with zero
        *******************************************************************************/
        division_by_zero(int numerator);

        /*!*****************************************************************************
        \brief
            Return error message telling user that there was an attempt with a
            divison_by_zero
        \return
            A message about an attempt of division_by_zero
        *******************************************************************************/
        virtual const char *what() const noexcept;

    private:
        std::string msg;
    };

    /*!*****************************************************************************
    \brief
        Constructor for division_by_zero
    \param [in] numerator
        numerator that is trying to divide with zero
    *******************************************************************************/
    division_by_zero::division_by_zero(int numerator)
    {
        std::ostringstream oss;
        std::cout << ".\n";
        oss << "Division by zero: " << numerator << " / 0!\n";
        msg = oss.str();
    }

    /*!*****************************************************************************
    \brief
        Return error message telling user that there was an attempt with a
        divison_by_zero
    \return
        A message about an attempt of division_by_zero
    *******************************************************************************/
    const char *division_by_zero::what() const noexcept
    {
        return msg.c_str();
    }

    /*!*****************************************************************************
    \brief
        Performs a simple division arithmetic passed in by function pointer
    \param [in] num
        Numerator
    \param [in] dem
        Denominator
    \param [in] func
        Function pointer that have some exception handling
    *******************************************************************************/
    template <typename F>
    void test1(int num, int dem, F func)
    {
        int res = 0;
        std::cout << "Calling function #1";
        if (!func(num, dem, res))
            throw division_by_zero(num);
        std::cout << "; result: " << res << "." << std::endl;
    }

    /*!*****************************************************************************
    \brief
        Performs a simple division arithmetic passed in by function pointer
    \param [in] num
        Numerator
    \param [in] dem
        Denominator
    \param [in] func
        Function pointer that have some exception handling
    *******************************************************************************/
    template <typename F>
    void test2(int num, int dem, F func)
    {
        std::pair<bool, int> p = func(num, dem);
        std::cout << "Calling function #2";
        if(!p.first)
            throw division_by_zero(num);
        std::cout << "; result: " << p.second << "." << std::endl;
    }

    /*!*****************************************************************************
    \brief
        Performs a simple division arithmetic passed in by function pointer
    \param [in] num
        Numerator
    \param [in] dem
        Denominator
    \param [in] func
        Function pointer that have some exception handling
    *******************************************************************************/
    template <typename F>
    void test3(int num, int dem, F func)
    {
        std::cout << "Calling function #3";
        int res = func(num, dem);
        if (!errno)
            throw division_by_zero(num);
        std::cout << "; result: " << res << "." << std::endl;
    }

    /*!*****************************************************************************
    \brief
        Performs a simple division arithmetic passed in by function pointer
    \param [in] num
        Numerator
    \param [in] dem
        Denominator
    \param [in] func
        Function pointer that have some exception handling
    *******************************************************************************/
    template <typename F>
    void test4(int num, int dem, F func)
    {
        std::cout << "Calling function #4";
        int res = func(num, dem);
        if (res <= std::numeric_limits<int>::min())
            throw division_by_zero(num);
        std::cout << "; result: " << res << "." << std::endl;
    }

    /*!*****************************************************************************
    \brief
        Performs a simple division arithmetic passed in by function pointer
    \param [in] num
        Numerator
    \param [in] dem
        Denominator
    \param [in] func
        Function pointer that have some exception handling
    *******************************************************************************/
    template <typename F>
    void test5(int num, int dem, F func)
    {
        std::cout << "Calling function #5";
        int res = 0;
        try
        {
            res = func(num, dem);
        }
        catch(division_by_zero const& dbz)
        {
            throw dbz;
        }
        std::cout << "; result: " << res << "." << std::endl;
    }
}