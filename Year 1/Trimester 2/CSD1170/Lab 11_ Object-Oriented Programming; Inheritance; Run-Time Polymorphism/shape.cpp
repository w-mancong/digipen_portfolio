/*!*****************************************************************************
\file shape.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 11
\date 01-04-2022
\brief
This file contains default definitions for all classes inheriting from 
Shape class.
*******************************************************************************/
#include "shape.hpp"
#include <iostream>
#include <sstream>

int Shape::count = 0;

/*!*********************************************************************************
    \brief
    Constructor for abstract Shape class that takes in a string
    \param [in] line:
    String containing all the relevant data for this shape
***********************************************************************************/
Shape::Shape(std::string &line)
{
    std::istringstream iss{ line };
    iss >> border >> fill;
    std::getline(iss, line);
    id = ++Shape::count;
}

/*!*********************************************************************************
    \brief
    Destructor function for Shape
***********************************************************************************/
Shape::~Shape(void) 
{
    --Shape::count;
}

/*!*********************************************************************************
    \brief
    Get the id of this shape
    \return 
    id of this shape
***********************************************************************************/
int Shape::getId(void) const
{
    return id;
}

/*!*********************************************************************************
    \brief
    Get the border color of this shape
    \return
    Border color of this shape
***********************************************************************************/
std::string Shape::getBorder() const
{
    return border;
}

/*!*********************************************************************************
    \brief
    Get the fill color of this shape
    \return
    Fill color of this shape
***********************************************************************************/
std::string Shape::getFill(void) const
{
    return fill;
}

/*!*********************************************************************************
    \brief
    Get the total number of shapes created
    \return
    Total number of shapes created
***********************************************************************************/
int Shape::getCount(void)
{
    return Shape::count;
}

/*!*********************************************************************************
    \brief
    Default function to print out the detail of shapes if this function is not
    overrided
***********************************************************************************/
void Shape::print_details(void) const
{
    std::cout << "Id = " << getId() << std::endl;
    std::cout << "Border = " << getBorder() << std::endl;
    std::cout << "Fill = " << getFill() << std::endl;
}

/*!*********************************************************************************
    \brief
    Default function to return the center point of this shape
    \return
    Center point: Point(0, 0)
***********************************************************************************/
Point Shape::getCenter(void) const
{
    return Point{0, 0};
}

/*!*********************************************************************************
    \brief
    Default function to return the area of this shape
    \return
    Area: 0.0
***********************************************************************************/
double Shape::getArea(void) const
{
    return 0.0;
}