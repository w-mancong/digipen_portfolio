/*!*****************************************************************************
\file point.cpp
\author Wong Man Cong   
\par DP email: w.mancong\@digipen.edu
\par Course: HLP2
\par Section: D
\par Lab 5
\date 12-02-2022
\brief
This file contains definitions for class Points and non-member operator 
overloaded functions
*******************************************************************************/

#include "point.hpp"  
#include <cmath>      

namespace
{
    const double PI = 3.141592653589793238;
    const double EPSILON = 0.00001;
}

namespace hlp2 
{
    /*!*****************************************************************************
    \brief
        Default constructor for class Point
    *******************************************************************************/
    Point::Point(void) : x(0.0), y(0.0) {}

    /*!*****************************************************************************
    \brief
        Constructor that takes in value for x and y

    \param [in] x
        x coordinate on the Cartesian Coordinate System 
    \param [in] y
        y coordinate on the Cartesian Coordinate System
    *******************************************************************************/
    Point::Point(double x, double y) : x(x), y(y) {}

    /*!*****************************************************************************
    \brief
        Copy Constructor

    \param [in] rhs
        Takes in another point to have it's value copied 
    *******************************************************************************/	
    Point::Point(Point const& rhs)
    {
        x = rhs.x;
        y = rhs.y;
    }

    /*!*****************************************************************************
    \brief
        Copy Assignment Operator overload
    
    \param [in] rhs
        Takes in another point to have it's value copied
    
    \return
        Reference to this class Point after copying value from rhs
    *******************************************************************************/
    Point& Point::operator=(Point const& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Accessor and mutator operator overload

    \param [in] index
        Between 0 and 1

    \return
        A reference to x value if index = 0, y value if index = 1
    *******************************************************************************/
    double& Point::operator[](size_t index)
    {
        return !index ? x : y;
    }

    /*!*****************************************************************************
    \brief
        Accessor operator overload
    
    \param [in] index
        Between 0 and 1

    \return
        x value if index = 0, y value if index = 1
    *******************************************************************************/
    const double& Point::operator[](size_t index) const
    {
        return !index ? x : y;
    }

    /*!*****************************************************************************
    \brief
        Adds the value of rhs's point with my lhs's point

    \param [in] rhs
        Another point class from the right hand side of the += operator
    
    \return 
        Reference to this class Point after adding the two point together
    *******************************************************************************/	   
    Point& Point::operator+=(Point const& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Adds this point class with a constant scalar

    \param [in] p
        A constant scalar to be added with this class Point

    \return
        Refernece to this class Point after add the point with the const scalar
    *******************************************************************************/
    Point& Point::operator+=(double const& p)
    {
        x += p; 
        y += p;
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Pre-increment operator
    
    \return
        Reference to this class Point after incrementing x and y by 1
    *******************************************************************************/
    Point& Point::operator++(void)
    {
        ++x, ++y;
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Post-increment operator

    \param 
        Just to differentiate between pre/post-increment

    \return
        rvalue of Point after incrementing the class Point x and y value by 1
    *******************************************************************************/
    const Point Point::operator++(int)
    {
        Point temp = *this;
        ++*this;
        return temp;
    }

    /*!*****************************************************************************
    \brief
        Post-decrement operator
    
    \return
        Reference to this class Point after decrementing x and y by 1
    *******************************************************************************/
    Point& Point::operator--(void)
    {
        --x, --y;
        return *this;
    }

    /*!*****************************************************************************
    \brief
        Post-decrement operator

    \param
        Just to differentiate between pre/post-increment

    \return 
        rvalue of Point after decrementing the class Point x and y value by 1
    *******************************************************************************/
    const Point Point::operator--(int)
    {
        Point temp = *this;
        --*this; 
        return temp;
    }

	/*!*****************************************************************************
	\brief
		Rotate Point by degree

	\param [in] lhs
		Point to be rotated
	\param [in] deg
		Angles to be rotated in degrees

	\return
		A new Point after rotating lhs by deg
    *******************************************************************************/
    Point operator%(Point const& lhs, double const& deg)
    {
        const double cos = std::cos(deg * PI / 180.0);
        const double sin = std::sin(deg * PI / 180.0);
        double x = lhs[0] * cos - lhs[1] * sin;
        double y = lhs[0] * sin + lhs[1] * cos;

        return Point((x > -EPSILON && x < EPSILON) ? 0.0 : x, (y > -EPSILON && y < EPSILON) ? 0.0 : y);
    }

	/*!*****************************************************************************
	\brief
		Distance between the two points

	\param [in] lhs
		1st Point
	\param [in] rhs
		2nd Point

	\return
		Distance between the two point
    *******************************************************************************/
	double operator/(Point const& lhs, Point const& rhs)
    {
        const double x = lhs[0] - rhs[0];
        const double y = lhs[1] - rhs[1];
        return sqrt(x * x + y * y);
    }

	/*!*****************************************************************************
	\brief
		Adding two points together

	\param [in] lhs
		1st point
	\param [in] rhs
		2nd point
	
	\return
		A new Point after adding lhs and rhs together
    *******************************************************************************/
	Point operator+(Point const& lhs, Point const& rhs)
    {
        return Point(lhs[0] + rhs[0], lhs[1] + rhs[1]);
    }

	/*!*****************************************************************************
	\brief
		Adding a point with a scalar
	
	\param [in] lhs
		Point class to be added
	\param [in] rhs
		Constant scalar to be added with point

	\return 
		A new Point after adding the Point and scalar together
    *******************************************************************************/
	Point operator+(Point const& lhs, double const& rhs)
    {
        return Point(lhs[0] + rhs, lhs[1] + rhs);
    }

	/*!*****************************************************************************
	\brief
		Adding a point with a scalar
	
	\param [in] lhs
		Constant scalar to be added with point
	\param [in] rhs
		Point class to be added

	\return 
		A new Point after adding the Point and scalar together
    *******************************************************************************/
	Point operator+(double const& lhs, Point const& rhs)
    {
        return Point(lhs + rhs[0], lhs + rhs[1]);
    }

	/*!*****************************************************************************
	\brief
		Subtracting two points together

	\param [in] lhs
		1st point
	\param [in] rhs
		2nd point
	
	\return
		A new Point after subtracting lhs from rhs
    *******************************************************************************/
	Point operator-(Point const& lhs, Point const& rhs)
    {
        return Point(lhs[0] - rhs[0], lhs[1] - rhs[1]);
    }

	/*!*****************************************************************************
	\brief
		Subtracting a point with a scalar
	
	\param [in] lhs
		Point class to be subtracted 
	\param [in] rhs
		Constant scalar to be subtracted with point

	\return 
		A new Point after subtracting the Point from scalar
    *******************************************************************************/	
	Point operator-(Point const& lhs, double const& rhs)
    {
        return Point(lhs[0] - rhs, lhs[1] - rhs);
    }

	/*!*****************************************************************************
	\brief
		Subtracting a point with a scalar
	
	\param [in] lhs
		Constant scalar to be subtracted with point
	\param [in] rhs
		Point class to be subtracted 

	\return 
		A new Point after subtracting the Point rom scalar
    *******************************************************************************/
	Point operator-(double const& lhs, Point const& rhs)
    {
        return Point(lhs - rhs[0], lhs - rhs[1]);
    }
	
	/*!*****************************************************************************
	\brief
		Negate the values in lhs

	\param [in] lhs
		Values to be negated

	\return
		A new Point after negating the values inside lhs
    *******************************************************************************/
	Point operator-(Point const& lhs)
    {
        return Point(-lhs[0], -lhs[1]);
    }

	/*!*****************************************************************************
	\brief
		Mid-point between lhs and rhs

	\param [in] lhs
		1st point
	\param [in] rhs
		2nd point

	\return
		The mid-point between lhs and rhs
    *******************************************************************************/
	Point operator^(Point const& lhs, Point const& rhs)
    {
        return Point((lhs[0] + rhs[0]) * 0.5, (lhs[1] + rhs[1]) * 0.5);
    }

	/*!*****************************************************************************
	\brief
		Dot product between the two points
	
	\param [in] lhs
		1st point
	\param [in] rhs
		2nd point
	
	\return
		Scalar quantity of the dot product between lhs and rhs
    *******************************************************************************/
	double operator*(Point const& lhs, Point const& rhs)
    {
        return lhs[0] * rhs[0] + lhs[1] * rhs[1];
    }

	/*!*****************************************************************************
	\brief
		Multiplying between a point and a scalar

	\param [in] lhs
		Point class to be multiplied 
	\param [in] rhs
		Constant scalar to be multiplied with point

	\return
		A new point class after multiplying point with scalar
    *******************************************************************************/
	Point operator*(Point const& lhs, double const& rhs)
    {
        return Point(lhs[0] * rhs, lhs[1] * rhs);
    }

	/*!*****************************************************************************
	\brief
		Multiplying between a point and a scalar

	\param [in] lhs
		Constant scalar to be multiplied with point
	\param [in] rhs
		Point class to be multiplied 

	\return
		A new point class after multiplying point with scalar
    *******************************************************************************/
	Point operator*(double const& lhs, Point const& rhs)
    {
        return Point(lhs * rhs[0], lhs * rhs[1]);
    }
	
	/*!*****************************************************************************
	\brief
		Overloaded output stream function
	
	\param [in, out] os
		Output stream to have Point p streamed into
	\param [in] p
		The point to have it's x and y value stream into os

	\return
		Reference to os
    *******************************************************************************/    
	std::ostream& operator<<(std::ostream& os, Point const& p)
    {
        return os << '(' << p[0] << ", " << p[1] << ')';
    }

	/*!*****************************************************************************
	\brief
		Overloaded input stream function

	\param [in] is
		Input stream to get Point p's value
	\param [in, out] p
		Storing the values streamed from is into p
	
	\return
		Reference to is
    *******************************************************************************/
	std::istream& operator>>(std::istream& is, Point& p)
    {
        return is >> p[0] >> p[1];
    }

} // end hlp2 namespace
