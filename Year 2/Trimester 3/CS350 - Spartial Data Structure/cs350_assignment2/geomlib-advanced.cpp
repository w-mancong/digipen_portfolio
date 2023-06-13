///////////////////////////////////////////////////////////////////////
// $Id$
//
// Geometric objects (Points, Vectors, Planes, ...) and operations.
//
// Gary Herron
//
// Copyright © 2007 DigiPen Institute of Technology
////////////////////////////////////////////////////////////////////////

#include "geomlib.h"
#include <vector>
#include <cassert>
#include <float.h> // FLT_EPSILON

////////////////////////////////////////////////////////////////////////
// Distance methods
////////////////////////////////////////////////////////////////////////

// Return the distance from a point to a line.
float Distance(const Point3D& point, const Line3D& line)
{
    throw Unimplemented();
}

// Return the distance from a point to a plane.
float Distance(const Point3D& point, const Plane3D& plane)
{
    throw Unimplemented();
}

////////////////////////////////////////////////////////////////////////
// Containment methods
////////////////////////////////////////////////////////////////////////

// Determines if point (known to be on a line) is contained within a segment.
bool Segment3D::contains(const Point3D& point) const
{   
    throw Unimplemented();
}

// Determines if point (known to be on a line) is contained within a ray.
bool Ray3D::contains(const Point3D& point, float *t) const
{
    throw Unimplemented();
}

// Determines if point is contained within a box.
bool Box3D::contains(const Point3D& point) const
{
    throw Unimplemented();
}

// Determines if point (known to be on a plane) is contained within a triangle.
bool Triangle3D::contains(const Point3D& point) const
{
    throw Unimplemented();
}

////////////////////////////////////////////////////////////////////////
// Intersects functions
// In the following Intersects function these rules apply:
//
// * Most test are to determine if a *unique* solution exists. (Or in
//   some cases up to two intersection points exist.)  Parallel
//   objects have either zero or infinitely many solutions and so
//   return false.
//
// * If a unique solution exists, a function value of true is
//   returned.  (Or in the cases where several solutions can exist,
//   the number of intersection parameters are returned.)
//
// * If a unique solution does exist, the calling program may provide
//   a memory location into which the intersection parameter can be
//   returned.  Such pointer may be NULL to indicate that the
//   intersection parameter is not to be returned.
//
////////////////////////////////////////////////////////////////////////

// Determines if 2D segments have a unique intersection.
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Segment2D& seg1, const Segment2D& seg2, float *rt)
{
    throw Unimplemented();
}

// Determines if 2D lines have a unique intersection.
// If true and rt is not NULL, returns intersection parameter.
// May not have been discussed in class.
bool Intersects(const Line2D& line1, const Line2D& line2, float *rt)
{
    throw Unimplemented();

}

// Determines if 3D line and plane have a unique intersection.  
// If true and t is not NULL, returns intersection parameter.
bool Intersects(const Line3D& line, const Plane3D& plane, float *rt)
{
    throw Unimplemented();
}

// Determines if 3D segment and plane have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Segment3D& seg, const Plane3D& plane, float *rt)
{
    throw Unimplemented();
}

// Determines if 3D segment and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Segment3D& seg, const Triangle3D& tri, float *rt)
{
    throw Unimplemented();
}

// Determines if 3D ray and sphere intersect.  
// If so and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray3D& ray, const Sphere3D& sphere, float *rt)
{
    throw Unimplemented();

}

// Determines if 3D ray and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray3D& ray, const Triangle3D& tri, float *rt)
{
    throw Unimplemented();

}

// Determines if 3D ray and AABB intersect.  
// If so and rt is not NULL, returns intersection parameter.
bool Intersects(const Ray3D& ray, const Box3D& box, float *rt)
{
    throw Unimplemented();
}

// Determines if 3D triangles intersect.  
// If parallel, returns false. (This may be considered misleading.)
// If true and rpoint is not NULL, returns two edge/triangle intersections.
int Intersects(const Triangle3D& tri1, const Triangle3D& tri2,
			   std::pair<Point3D, Point3D> *rpoints)
{
    throw Unimplemented();

}

////////////////////////////////////////////////////////////////////////
// Geometric relationships
////////////////////////////////////////////////////////////////////////

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Line3D& line1, const Line3D& line2)
{
    throw Unimplemented();
}

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Line3D& line, const Plane3D& plane)
{
    throw Unimplemented();
}

// Compute angle between two geometric entities (in radians;  use acos)
float AngleBetween(const Plane3D& plane1, const Plane3D& plane2)
{
    throw Unimplemented();
}

// Determine if two vectors are parallel.
bool Parallel(const Vector3D& v1, const Vector3D& v2)
{
    throw Unimplemented();
}

bool Perpendicular(const Vector3D& v1, const Vector3D& v2)
{
    throw Unimplemented();
}

// Determine if two lines are coplanar
bool Coplanar(const Line3D& line1, const Line3D& line2)
{
    throw Unimplemented();
}

// Determine if two geometric entities are parallel.
bool Parallel(const Line3D& line1, const Line3D& line2)
{
    throw Unimplemented();
}

// Determine if two geometric entities are parallel.
bool Parallel(const Line3D& line, const Plane3D& plane)
{
    throw Unimplemented();
}

// Determine if two geometric entities are parallel.
bool Parallel(const Plane3D& plane1, const Plane3D& plane2)
{
    throw Unimplemented();
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Line3D& line1, const Line3D& line2)
{
    throw Unimplemented();
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Line3D& line, const Plane3D& plane)
{
    throw Unimplemented();
}

// Determine if two geometric entities are perpendicular.
bool Perpendicular(const Plane3D& plane1, const Plane3D& plane2)
{
    throw Unimplemented();
}
