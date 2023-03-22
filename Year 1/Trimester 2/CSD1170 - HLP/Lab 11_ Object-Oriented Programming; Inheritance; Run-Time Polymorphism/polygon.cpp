/*!*****************************************************************************
\file polygon.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 11
\date 01-04-2022
\brief
This file contains overrided functionalities that is unique to Polygon class
*******************************************************************************/
#include "polygon.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

int Polygon::count = 0;

/*!*********************************************************************************
    \brief
    Constructor for Polygon class that takes in a string
    \param [in] line:
    String containing all the relevant data for this polygon class
***********************************************************************************/
Polygon::Polygon(std::string &line) : Shape{ line }
{
    std::istringstream iss{ line }; std::string buffer{};
    while (!iss.eof())
    {
        iss >> buffer;
        size_t comma_index = buffer.find_first_of(',');
        std::string x_num = buffer.substr(0, comma_index), y_num = buffer.substr(comma_index + 1);
        vertices.push_back(Point{ std::stoi(x_num), std::stoi(y_num) });
    }
    ++Polygon::count;
}

/*!*********************************************************************************
    \brief
    Destructor function for Polygon class
***********************************************************************************/
Polygon::~Polygon(void)
{
    --Polygon::count;
}

/*!*********************************************************************************
    \brief
    Get the vector containing all the vertices of the Polygon
    \return
    Vector containing all the vertices of the Polygon
***********************************************************************************/
std::vector<Point> &Polygon::getVertices(void)
{
    return vertices;
}

/*!*********************************************************************************
    \brief
    Get the total number of Polygon created
    \return
    Total number of Polygon created
***********************************************************************************/
int Polygon::getCount(void)
{
    return Polygon::count;
}

/*!*********************************************************************************
    \brief
    Printing details relevant to Polygon
***********************************************************************************/
void Polygon::print_details(void) const
{
    std::cout << "Id = " << getId() << std::endl;
    std::cout << "Border = " << getBorder() << std::endl;
    std::cout << "Fill = " << getFill() << std::endl;
    std::cout << "Type = Polygon Shape" << std::endl;
    std::cout << "Vertices = ";
    for (Point const &p : vertices)
        std::cout << p.x << "," << p.y << ' ';
    std::cout << std::endl << std::endl;
}

/*!*********************************************************************************
    \brief
    Return the center point of Polygon
    \return
    Center point of Polygon
***********************************************************************************/
Point Polygon::getCenter(void) const
{
    int cen_x = 0, cen_y = 0;
    for(Point const& p : vertices)
        cen_x += p.x, cen_y += p.y;
    return Point{ cen_x / static_cast<int>(vertices.size()), cen_y / static_cast<int>(vertices.size()) };
}

/*!*********************************************************************************
    \brief
    Calculate and return the area of Polygon
    \return 
    Area of the Polygon
***********************************************************************************/
double Polygon::getArea(void) const
{
    std::vector<Point> p{vertices}; p.push_back(vertices[0]);
    int lhs = 0, rhs = 0;
    for (size_t i = 0; i < p.size() - 1; ++i)
    {
        lhs += p[i].x * p[i + 1].y;
        rhs += p[i + 1].x * p[i].y;
    }
    return 0.5 * (static_cast<double>(lhs - rhs));
}