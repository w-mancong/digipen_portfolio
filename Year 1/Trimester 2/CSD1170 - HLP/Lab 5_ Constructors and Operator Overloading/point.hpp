/*!*****************************************************************************
\file point.hpp
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

////////////////////////////////////////////////////////////////////////////////
#ifndef POINT_HPP
#define POINT_HPP
////////////////////////////////////////////////////////////////////////////////

#include <iostream> // istream, ostream

namespace hlp2 
{	
	class Point 
	{
	public:
		/*!*****************************************************************************
		\brief
    	    Default constructor for class Point
    	*******************************************************************************/
		Point(void);

		/*!*****************************************************************************
		\brief
			Constructor that takes in value for x and y

		\param [in] x
			x coordinate on the Cartesian Coordinate System 
		\param [in] y
			y coordinate on the Cartesian Coordinate System
    	*******************************************************************************/
		Point(double x, double y);
		
		/*!*****************************************************************************
		\brief
			Copy Constructor

		\param [in] rhs
			Takes in another point to have it's value copied 
    	*******************************************************************************/		
		Point(Point const& rhs); 
		
		/*!*****************************************************************************
		\brief
			Copy Assignment Operator overload
		
		\param [in] rhs
			Takes in another point to have it's value copied
		
		\return
			Reference to this class Point after copying value from rhs
    	*******************************************************************************/
		Point& operator=(Point const& rhs);
		
		/*!*****************************************************************************
		\brief
			Accessor and mutator operator overload

		\param [in] index
			Between 0 and 1

		\return
			A reference to x value if index = 0, y value if index = 1
    	*******************************************************************************/		
		double& operator[](size_t index);

		/*!*****************************************************************************
		\brief
			Accessor operator overload
		
		\param [in] index
			Between 0 and 1

		\return
			x value if index = 0, y value if index = 1
    	*******************************************************************************/		
		const double& operator[](size_t index) const;
		
		/*!*****************************************************************************
		\brief
			Adds the value of rhs's point with my lhs's point

		\param [in] rhs
			Another point class from the right hand side of the += operator
		
		\return 
			Reference to this class Point after adding the two point together
    	*******************************************************************************/		
		Point& operator+=(Point const& rhs);
		
		/*!*****************************************************************************
		\brief
			Adds this point class with a constant scalar

		\param [in] p
			A constant scalar to be added with this class Point

		\return
			Refernece to this class Point after add the point with the const scalar
    	*******************************************************************************/		
		Point& operator+=(double const& p);

		/*!*****************************************************************************
		\brief
			Pre-increment operator
		
		\return
			Reference to this class Point after incrementing x and y by 1
    	*******************************************************************************/
		Point& operator++(void);

		/*!*****************************************************************************
		\brief
			Post-increment operator

		\param 
			Just to differentiate between pre/post-increment

		\return
			rvalue of Point after incrementing the class Point x and y value by 1
    	*******************************************************************************/		
		const Point operator++(int);

		/*!*****************************************************************************
		\brief
			Post-decrement operator
		
		\return
			Reference to this class Point after decrementing x and y by 1
    	*******************************************************************************/
		Point& operator--(void);

		/*!*****************************************************************************
		\brief
			Post-decrement operator

		\param
			Just to differentiate between pre/post-increment

		\return 
			rvalue of Point after decrementing the class Point x and y value by 1
    	*******************************************************************************/		
		const Point operator--(int);
		
	private:
		double x; // The x-coordinate of a Point
		double y; // The y-coordinate of a Point
	};
	

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
	Point operator%(Point const& lhs, double const& deg);

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
	double operator/(Point const& lhs, Point const& rhs);

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
	Point operator+(Point const&  lhs, Point const& rhs);

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
	Point operator+(Point const&  lhs, double const& rhs);

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
	Point operator+(double const& lhs, Point const& rhs);

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
	Point operator-(Point const&  lhs, Point const& rhs);

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
	Point operator-(Point const&  lhs, double const& rhs);

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
	Point operator-(double const& lhs, Point const& rhs);
	
	/*!*****************************************************************************
	\brief
		Negate the values in lhs

	\param [in] lhs
		Values to be negated

	\return
		A new Point after negating the values inside lhs
    *******************************************************************************/	
	Point operator-(Point const& lhs);

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
	Point operator^(Point const& lhs, Point const& rhs);

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
	double operator*(Point const& lhs, Point const& rhs);

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
	Point operator*(Point const& lhs, double const& rhs);

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
	Point operator*(double const& lhs, Point const& rhs);
	
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
	std::ostream& operator<<(std::ostream& os, Point const& p);

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
	std::istream& operator>>(std::istream& is, Point& p);
}

#endif // end POINT_HPP
////////////////////////////////////////////////////////////////////////////////
