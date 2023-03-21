/*!*****************************************************************************
\file ellipse.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 11
\date 01-04-2022
\brief
This file contains overrided functionalities that is unique to Ellipse class
*******************************************************************************/
#include "ellipse.hpp"
#include <iostream>
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>

int Ellipse::count = 0;

/*!*********************************************************************************
    \brief
    Constructor for Ellipse class that takes in a string
    \param [in] line:
    String containing all the relevant data for this ellipse class
***********************************************************************************/
Ellipse::Ellipse(std::string &line) : Shape{ line }
{
    std::istringstream iss{ line }; std::string buffer{};
    iss >> buffer >> a >> b;
    size_t comma_index = buffer.find_first_of(',');
    center = Point{ std::stoi(buffer.substr(0, comma_index)), std::stoi(buffer.substr(comma_index + 1)) };
    ++Ellipse::count;
}

/*!*********************************************************************************
    \brief
    Destructor function for Ellipse
***********************************************************************************/
Ellipse::~Ellipse(void)
{
    --Ellipse::count;
}

/*!*********************************************************************************
    \brief
    Get the length of side A
    \return
    Length of side A
***********************************************************************************/
int Ellipse::getA(void) const
{
    return a;
}

/*!*********************************************************************************
    \brief
    Get the length of side B
    \return
    Length of side B
***********************************************************************************/
int Ellipse::getB(void) const
{
    return b;
}

/*!*********************************************************************************
    \brief
    Get the total number of Ellipse created
    \return 
    Total number of Ellipse created
***********************************************************************************/
int Ellipse::getCount(void)
{
    return Ellipse::count;
}

/*!*********************************************************************************
    \brief
    Printing details relevant to Ellipse
***********************************************************************************/
void Ellipse::print_details(void) const
{
    std::cout << "Id = " << getId() << std::endl;
    std::cout << "Border = " << getBorder() << std::endl;
    std::cout << "Fill = " << getFill() << std::endl;
    std::cout << "Type = Ellipse Shape" << std::endl;
    std::cout << "Center = " << center.x << "," << center.y << std::endl;
    std::cout << "a-length = " << a << std::endl;
    std::cout << "b-length = " << b << std::endl << std::endl;
}

/*!*********************************************************************************
    \brief
    Get the center point of Ellipse
    \return
    Center point of the Ellipse
***********************************************************************************/
Point Ellipse::getCenter(void) const
{
    return center;
}

/*!*********************************************************************************
    \brief
    Calculate and return the area of the Ellipse
    \return
    Area of the Ellipse
***********************************************************************************/
double Ellipse::getArea(void) const
{
    return a * b * M_PI;
}