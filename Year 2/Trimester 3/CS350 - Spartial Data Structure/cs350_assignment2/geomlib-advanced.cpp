/*!*****************************************************************************
\file
\author:    Wong Man Cong
\co-author: Gary Herron
\par DP email: w.mancong\@digipen.edu
               gherron@digipen.edu
\par Course: CS350
\par Section: A
\par Assignment 1
\date 19th May 2023
\brief
Geometric objects (Points, Vectors, Planes, ...) and operations.

Copyright © 2007 DigiPen Institute of Technology
*******************************************************************************/

#include "geomlib.h"
#include <vector>
#include <cassert>
#include <float.h> // FLT_EPSILON

namespace
{
    Vector3D const zero(0.0f, 0.0f, 0.0f);

    /*!*****************************************************************************
        \brief Checks if the two floats a and b are equivalent

        \param [in] a, b: floats to compare

        \return True if both floats are equivalent, else false
    *******************************************************************************/
    bool Equal(float a, float b)
    {
        return fabs(a - b) <= FLT_EPSILON;
    }
}

/*!*****************************************************************************
    \brief Calculate the distance between a point and line

    \param [in] point: Point in space
    \param [in] line: Line in space

    \return Distance between the two geometric entities
*******************************************************************************/
float Distance(const Point3D& point, const Line3D& line)
{
    return glm::length( glm::cross(line.vector, point - line.point) ) / glm::length(line.vector);
}

/*!*****************************************************************************
    \brief Calculate the distance between a point and plane

    \param [in] point: Point in space
    \param [in] plane: Plane in space

    \return Distance between the two geometric entities
*******************************************************************************/
float Distance(const Point3D& point, const Plane3D& plane)
{
    return glm::abs( plane.Evaluate(point) / glm::length(plane.normal()) );
}

/*!*****************************************************************************
    \brief Determines if point (known to be on a line) is contained within 
           a segment

    \param [in] point: Point in space

    \return True if point is within the line segment, else false
*******************************************************************************/
bool Segment3D::contains(const Point3D& point) const
{   
    auto check = [](float p1, float p2, float p)
    {
        if (p1 != p2)
        {
            return ((p2 - p1 > 0.0f) && (p1 <= p && p <= p2))
                || ((p2 - p1 < 0.0f) && (p1 >= p && p >= p2));
        }
        return false;
    };
    return check(point1.x, point2.x, point.x) || check(point1.y, point2.y, point.y) || check(point1.z, point2.z, point.z);
}

/*!*****************************************************************************
    \brief Determines if point (known to be on a line) is contained within
           a ray

    \param [in] point: Point in space
    \param [out] t: Scalar value of the intersection between the ray and the point

    \return True if point is within the ray, else false
*******************************************************************************/
bool Ray3D::contains(const Point3D& point, float *t) const
{
    if (t)
        *t = (point.x - origin.x) / direction.x;
    return glm::dot((point - origin), direction) >= 0.0f;   
}

/*!*****************************************************************************
    \brief Determines if point is contained with a box

    \param [in] point: Point in space

    \return True if point is within the box, else false
*******************************************************************************/
bool Box3D::contains(const Point3D& point) const
{
    float x1 = center.x + extents.x, x2 = center.x - extents.x,
          y1 = center.y + extents.y, y2 = center.y - extents.y,
          z1 = center.y + extents.z, z2 = center.z - extents.z;
    float x = point.x, y = point.y, z = point.z;

    return x <= x1 && x >= x2 && y <= y1 && y >= y2 && z <= z1 && z >= z2;
}

/*!*****************************************************************************
    \brief Determines if point (known to be on a plane) is contained within
           a triangle

    \param [in] point: Point in space

    \return True if point is within the triangle, else false
*******************************************************************************/
bool Triangle3D::contains(const Point3D& point) const
{
    Vector3D const n = glm::cross(points[1] - points[0], points[2] - points[0]);
    auto check = [&n, &point](Point3D const& a, Point3D const& b)
    {
        return glm::dot( point - a, glm::cross( n, b - a ) ) >= 0.0f;
    };
    return check(points[0], points[1]) && check(points[1], points[2]) && check(points[2], points[0]);
}

// Determines if 2D segments have a unique intersection.
// If true and rt is not NULL, returns intersection parameter.
/*!*****************************************************************************
    \brief Determines if 2D segments have a unique intersection

    \param [in] seg1, seg2: Line segment in space
    \param [out] rt: Scalar value of the intersection between the two segments

    \return True if both line segments intersects, else false
*******************************************************************************/
bool Intersects(const Segment2D& seg1, const Segment2D& seg2, float *rt)
{
    return Intersects(Line2D(seg1.point1, seg1.point2 - seg1.point1), 
                      Line2D(seg2.point1, seg2.point2 - seg2.point1), rt);
}

/*!*****************************************************************************
    \brief Determines if 2D lines have a unique intersection

    \param [in] line1, line2: Line in space
    \param [out] rt: Scalar value of the intersection between the two line

    \return True if both line intersects, else false
*******************************************************************************/
bool Intersects(const Line2D& line1, const Line2D& line2, float *rt)
{
    float det = line1.vector.x * line2.vector.y - line1.vector.y * line2.vector.x;

    if (abs(det) > FLT_EPSILON)
    {
        if (rt)
        {
            Vector2D diff = line2.point - line1.point;
            float t1 = (diff.x * line2.vector.y - diff.y * line2.vector.x) / det;
            float t2 = (diff.x * line1.vector.y - diff.y * line1.vector.x) / det;
            *rt = t1;
        }
        return true;
    }
    return false;
}

/*!*****************************************************************************
    \brief Determines if 3D line and plane have a unique intersection

    \param [in] line: Line in space
    \param [in] plane: Plane in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the line and plane intersects, else false
*******************************************************************************/
bool Intersects(const Line3D& line, const Plane3D& plane, float *rt)
{
    float dp = glm::dot(line.vector, plane.normal());

    if (abs(dp) > FLT_EPSILON)
    {
        if (rt)
        {
            float  dist = plane.Evaluate(line.point);
            *rt = -dist / dp;
        }
        return true;
    }

    return false;
}

/*!*****************************************************************************
    \brief Determiens if 3D segment and plane have a unique intersection

    \param [in] seg: Line segment in space
    \param [in] plane: Plane in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the line segment and plane intersects, else false
*******************************************************************************/
bool Intersects(const Segment3D& seg, const Plane3D& plane, float *rt)
{
    return Intersects(Line3D(seg.point1, seg.point2 - seg.point1), plane, rt);
}

// Determines if 3D segment and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
/*!*****************************************************************************
    \brief Determines if 3D segment and triangle have a unique intersection

    \param [in] seg: Line segment in space
    \param [in] tri: Triangle in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the line segment and triangle intersects, else false
*******************************************************************************/
bool Intersects(const Segment3D& seg, const Triangle3D& tri, float *rt)
{
    return Intersects(Ray3D(seg.point1, seg.point2 - seg.point1), tri, rt);
}

/*!*****************************************************************************
    \brief Determines if 3D ray and sphere intersects

    \param [in] ray: Ray in space
    \param [in] sphere: Sphere in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the ray and sphere intersects, else false
*******************************************************************************/
bool Intersects(const Ray3D& ray, const Sphere3D& sphere, float *rt)
{
    // Compute the direction from the ray's origin to the sphere's center
    Vector3D sphereToRay = ray.origin - sphere.center;

    // Compute the coefficients of the quadratic equation
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(ray.direction, sphereToRay);
    float c = dot(sphereToRay, sphereToRay) - sphere.radius * sphere.radius;

    // Compute the discriminant
    float discriminant = b * b - 4.0f * a * c;

    // Check if the ray intersects the sphere
    if (discriminant < 0.0f)
        return false;

    // Compute the intersection parameter(s)
    float sqrtDiscriminant = std::sqrt(discriminant);
    float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
    float t2 = (-b + sqrtDiscriminant) / (2.0f * a);

    // Check if the intersection point is in front of the ray
    if (t1 < FLT_EPSILON && t2 < FLT_EPSILON)
        return false;

    // Store the intersection parameter if requested
    if (rt)
    {
        if (t1 >= FLT_EPSILON && t2 >= FLT_EPSILON)
            *rt = std::min(t1, t2);
        else if (t1 >= FLT_EPSILON)
            *rt = t1;
        else
            *rt = t2;
    }

    return true;
}

// Determines if 3D ray and triangle have a unique intersection.  
// If true and rt is not NULL, returns intersection parameter.
/*!*****************************************************************************
    \brief Determines if 3D ray and triangle have a unique intersection

    \param [in] ray: Ray in space
    \param [in] tri: Triangle in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the ray and triangle intersects, else false
*******************************************************************************/
bool Intersects(const Ray3D& ray, const Triangle3D& tri, float *rt)
{
    // Compute the triangle's normal
    Vector3D N = glm::cross(tri[2] - tri[0], tri[1] - tri[0]);

    // Check if the ray is parallel to the triangle's plane
    float dp = dot(ray.direction, N);
    if (std::abs(dp) < FLT_EPSILON)
        return false;

    // Compute the distance from the ray's origin to the triangle's plane
    Point3D pointOnPlane = tri[0];
    float t = dot(pointOnPlane - ray.origin, N) / dp;

    // Check if the intersection point is behind the ray
    if (t < 0.0f)
        return false;

    // Compute the intersection point on the triangle's plane
    Point3D I = ray.origin + t * ray.direction;

    // Check if the intersection point is inside the triangle
    if (!tri.contains(I))
        return false;

    // Store the intersection parameter if requested
    if (rt)
        *rt = t;

    return true;
}

/*!*****************************************************************************
    \brief Determines if 3D ray and AABB intersects
    
    \param [in] ray: Ray in space
    \param [in] box: Box in space
    \param [out] rt: Scalar value of the intersection between the two geometric
                     entities

    \return True if the ray and box intersects, else false
*******************************************************************************/
bool Intersects(const Ray3D& ray, const Box3D& box, float *rt)
{
    float t0 = -std::numeric_limits<float>::infinity();
    float t1 = std::numeric_limits<float>::infinity();

    Point3D min{ box.center.x - box.extents.x, box.center.y - box.extents.y ,box.center.z - box.extents.z };
    Point3D max{ box.center.x + box.extents.x, box.center.y + box.extents.y ,box.center.z + box.extents.z };
    for (size_t i = 0; i < 3; ++i)
    {
        if (ray.direction[i] == 0.0f)
        {
            if (ray.origin[i] < min[i] || ray.origin[i] > max[i])
                return false;
        }
        else
        {
            float invDirection = 1.0f / ray.direction[i];
            float s0 = (min[i] - ray.origin[i]) * invDirection;
            float s1 = (max[i] - ray.origin[i]) * invDirection;

            if (s0 > s1)
                std::swap(s0, s1);

            t0 = std::max(t0, s0);
            t1 = std::min(t1, s1);

            if (t0 > t1)
                return false;
        }
    }

    if (t1 < 0.0f)
        return false;
    else if (t0 < 0.0f)
    {
        if (rt)
            *rt = t1;
        return true;
    }
    else
    {
        if (rt)
            *rt = t0;
        return true;
    }
    return false;
}

/*!*****************************************************************************
    \brief Determines if 3D triangles intersect

    \param [in] tri1, tri2: Triangles in space
    \param [out] rpoints: Intersection points of the two triangles

    \return Number of intersections between the triangles
*******************************************************************************/
int Intersects(const Triangle3D& tri1, const Triangle3D& tri2, std::pair<Point3D, Point3D> *rpoints)
{
    int numOfIntersections = 0;
    Point3D const MIN_POINT = { FLT_MIN, FLT_MIN, FLT_MIN };
    if(rpoints)
        rpoints->first = rpoints->second = MIN_POINT;

    auto GetIntersectionPoint = [&MIN_POINT, &rpoints](Point3D p, Vector3D v, float t)
    {
        Point3D q = p + t * v;  // intersection point
        if (!rpoints)
            return;
        if (rpoints->first == MIN_POINT)
            rpoints->first = q;
        else
            rpoints->second = q;
    };

    auto check = [&numOfIntersections, &GetIntersectionPoint](Triangle3D const& t1, Triangle3D const& t2)
    {
        for (size_t i{}; i < 3; ++i)
        {
            float t{};
            if (Intersects(Segment3D(t1[i], t1[(i + 1) % 3]), t2, &t))
            {
                GetIntersectionPoint(t1[i], t1[(i + 1) % 3] - t1[i], t);
                ++numOfIntersections;
            }
        }
    };

    check(tri1, tri2); check(tri2, tri1);

    return numOfIntersections;
}

/*!*****************************************************************************
    \brief Compute angle between two lines

    \param [in] line1, line2: Lines in space

    \return Angle between the two lines in radians
*******************************************************************************/
float AngleBetween(const Line3D& line1, const Line3D& line2)
{
    return glm::acos( ( glm::dot(line1.vector, line2.vector) ) / 
             ( glm::length(line1.vector) * glm::length(line2.vector) ) );
}

/*!*****************************************************************************
    \brief Compute the angle between a line and a plane

    \param [in] line: Line in space
    \param [in] plane: Plane in space

    \return Angle between the line and plane in radians
*******************************************************************************/
float AngleBetween(const Line3D& line, const Plane3D& plane)
{
    return PI * 0.5f - ( glm::acos( glm::dot(plane.normal(), line.vector) / ( glm::length(plane.normal()) * glm::length(line.vector) ) ) );
}

/*!*****************************************************************************
    \brief Compute the angle between two planes

    \param [in] plane1, plane2: Planes in space

    \return Angle between the two planes in radians
*******************************************************************************/
float AngleBetween(const Plane3D& plane1, const Plane3D& plane2)
{
    return glm::acos( ( glm::dot(plane1.normal(), plane2.normal() ) / 
        ( glm::length( plane1.normal() * glm::length( plane2.normal() ) ) ) ) );
}

/*!*****************************************************************************
    \brief Determines if two vectors are parallel

    \param [in] v1, v2: Vectors in space
    
    \return True if both vectors are parallel, else false
*******************************************************************************/
bool Parallel(const Vector3D& v1, const Vector3D& v2)
{
    return glm::cross(v1, v2) == zero;
}

/*!*****************************************************************************
    \brief Determines if two vector are perpendicular

    \param [in] v1, v2: Vectors in space

    \return True if both vectors are perpendicular, else false
*******************************************************************************/
bool Perpendicular(const Vector3D& v1, const Vector3D& v2)
{
    return Equal(glm::dot(v1, v2), 0.0f);
}

/*!*****************************************************************************
    \brief Determines if two lines are coplanar

    \param [in] line1, line2: Lines in space

    \return True if both lines are coplanar, else false
*******************************************************************************/
bool Coplanar(const Line3D& line1, const Line3D& line2)
{
    return Equal(glm::dot( line2.point - line1.point, glm::cross(line1.vector, line2.vector) ), 0.0f);
}

/*!*****************************************************************************
    \brief Determines if two lines are parallel

    \param [in] line1, line2: Lines in space

    \return True if both lines are parallel, else false
*******************************************************************************/
bool Parallel(const Line3D& line1, const Line3D& line2)
{
    return Parallel(line1.vector, line2.vector);
}

/*!*****************************************************************************
    \brief Determines if the line and plane are parallel

    \param [in] line: Line in space
    \param [in] plane: Plane in space

    \return True if the line and plane are parallel, else false
*******************************************************************************/
bool Parallel(const Line3D& line, const Plane3D& plane)
{
    return Perpendicular(plane.normal(), line.vector);
}

/*!*****************************************************************************
    \brief Determines is two planes are parallel

    \param [in] plane1, plane2: Planes in space

    \return True if both planes are parallel, else false
*******************************************************************************/
bool Parallel(const Plane3D& plane1, const Plane3D& plane2)
{
    return Parallel(plane1.normal(), plane2.normal());
}

/*!*****************************************************************************
    \brief Determines if two lines are perpendicular

    \param [in] line1, line2: Lines in space

    \return True if both lines are perpendicular, else false
*******************************************************************************/
bool Perpendicular(const Line3D& line1, const Line3D& line2)
{
    return Perpendicular(line1.vector, line2.vector);
}

/*!*****************************************************************************
    \brief Determnines if a line and a plane is perpendicular

    \param [in] line: Line in space
    \param [in] plane: Plane in space

    \return True if the line and plane is perpendicular, else false
*******************************************************************************/
bool Perpendicular(const Line3D& line, const Plane3D& plane)
{
    return Parallel(plane.normal(), line.vector);
}

/*!*****************************************************************************
    \brief Determines if two planes are perpendicular

    \param [in] plane1, plane2: Planes in space

    \return True if both planes are perpendicular, else false
*******************************************************************************/
bool Perpendicular(const Plane3D& plane1, const Plane3D& plane2)
{
    return Perpendicular(plane1.normal(), plane2.normal());
}
