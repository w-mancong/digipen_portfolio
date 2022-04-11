/*!*****************************************************************************
\file process.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 11
\date 01-04-2022
\brief
This file contains function definition to read from file, print shapes information,
record of all shapes and to deallocate memory from the heap
*******************************************************************************/

#include "process.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

/*!*********************************************************************************
    \brief
    Read from file stream and instantiate objects of the particular type
    \param [in] ifs:
    Input file stream containing all the data relevant to creating Shapes
    \param [out] vs:
    Vector to store all of the shapes created
    \param [out] ves:
    Vector to store all of the ellipse created
    \param [out] vps:
    Vector to store all of the polygon created
***********************************************************************************/
void parse_file(std::ifstream &ifs, std::vector<Shape*>& vs,
                std::vector<Shape*>& ves, std::vector<Shape*>& vps) 
{
  std::string buffer{};
  while(std::getline(ifs, buffer))
  {
    Shape *shape = nullptr; char const c = buffer[0];
    buffer = buffer.substr(2);
    switch (c)
    {
      case 'E':
      {
        shape = new Ellipse(buffer);
        ves.push_back(shape);
        break;
      }
      case 'P':
      {
        shape = new Polygon(buffer);
        vps.push_back(shape);
        break;
      }
    }
    vs.push_back(shape);
  }
}

/*!*********************************************************************************
    \brief
    Print details of all the shapes
    \param [in] vs:
    Vector containing all the shapes
***********************************************************************************/
void print_records(std::vector<Shape*> const& vs) 
{
  for(Shape* ps : vs)
    ps->print_details();
}

/*!*********************************************************************************
    \brief
    Print the stats of all the shapes created
    \param [in] vs:
    Vector containing all the shapes
***********************************************************************************/
void print_stats(std::vector<Shape*> const& vs) 
{

  std::vector<Shape*> tmp{vs};
  std::sort(tmp.begin(), tmp.end(), [](Shape const* lhs, Shape const* rhs)
  { 
    return lhs->getArea() < rhs->getArea(); 
  });
  double area{};
  for(Shape* ps : vs)
    area += ps->getArea();
  std::cout << "Number of shapes = " << vs.size() << std::endl;
  std::cout << "The mean of the areas = " << area / vs.size() << std::endl;
  std::cout << "The sorted list of shapes (id,center,area) in ascending order of areas:" << std::endl;
  for (Shape *ps : tmp)
    std::cout << ps->getId() << "," << ps->getCenter().x << "," << ps->getCenter().y << "," << ps->getArea() << std::endl;
}

/*!*********************************************************************************
    \brief
    Function to deallocate all the memory inside the vector
    \param [in] vs:
    Vector containing all the shapes (Deallocating memory inside this vector)
    \param [in] ves:
    Vector containing all the ellipse
    \param [in] vps:
    Vector containing all the polygon
***********************************************************************************/
void cleanup(std::vector<Shape*>& vs, std::vector<Shape*>& ves, std::vector<Shape*>& vps) 
{
  for (Shape *ps : vs)
    delete ps;

  vs.clear();
  ves.clear();
  vps.clear();
}
