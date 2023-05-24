///////////////////////////////////////////////////////////////////////
// $Id: Geometry.cpp 972 2007-05-29 23:42:53Z gherron $
//
// Unit tests of the geometry library.
//
// Gary Herron
//
// Copyright © 2007 DigiPen Institute of Technology
////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <iostream>
#include <fstream>

#include "geomlib.h"

using namespace std;

const float epsilon = 1e-3f;

int failures = 0;
int completed = 0;
int missing = 0;
int expected = 128;
#define AssertHere _AssertHere(__FILE__,__LINE__)
#define AssertNotHere(msg) _AssertNotHere(msg, __FILE__,__LINE__)
#define AssertEq(A,B,msg) _AssertEq(msg, A, B, __FILE__, __LINE__)

bool _AssertHere(const char* file, int line)
{
    completed++;
    std::cout << "Passed "  << file << ":" << line << std::endl;
    return true;
}

bool _AssertNotHere(const char* name, const char* file, int line)
{
    completed++;
    if (!failures++) std::cout << "Failed tests:" <<  std::endl;
    std::cout << "  " << name << "  At line "  << file << ":" << line << std::endl;
    return false;
}

bool _AssertEq(const char* name, const float A, const float B,
               const char* file, int line)
{
    completed++;
    if (fabs(A-B) > epsilon) {
        if (!failures++) std::cout << "Failed tests:" <<  std::endl;
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const Vector2D A, const Vector2D B,
               const char* file, int line)
{
    completed++;
    if (fabs(A[0]-B[0]) > epsilon ||
        fabs(A[1]-B[1]) > epsilon) {
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        failures++;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const Vector3D A, const Vector3D B,
               const char* file, int line)
{
    completed++;
    if (fabs(A[0]-B[0]) > epsilon ||
        fabs(A[1]-B[1]) > epsilon ||
        fabs(A[2]-B[2]) > epsilon)
    {
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        failures++;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const Vector4D A, const Vector4D B,
               const char* file, int line)
{
    completed++;
    if (fabs(A[0]-B[0]) > epsilon ||
        fabs(A[1]-B[1]) > epsilon ||
        fabs(A[2]-B[2]) > epsilon ||
        fabs(A[3]-B[3]) > epsilon) {
        std::cout << "  " << name << "  At line " << file << ":" << line << std::endl;
        failures++;
        return false; }
    else {
        //std::cout << "Passed " << file << ":" << line << std::endl;
        return true; }
}

bool _AssertEq(const char* name, const Matrix4x4 A, const Matrix4x4 B,
               const char* file, int line)
{
    completed++;
    return A==B;
}

void PointLineDistance()
{

    Point3D  P(9.0f, 6.0f, 5.0f);
    Vector3D V(-83.0f, 52.0f, 344.65f);
    Line3D line(P, V);
    Vector3D N = normalize(cross(V, Vector3D(83.0f, 52.0f, 344.65f)));

    AssertEq(Distance(P,line), 0,
             "Distance of P to line<P,V>.");
    AssertEq(Distance(P+V,line), 0,
             "Distance of P+V to line<P,V>");

    for (float d=1;     d<=2;  d+=0.5) {
        Point3D Q0 = P + d*N;
        AssertEq(Distance(Q0,line), d,
                 "Distance of P+d*N to line<P,V>");
        Point3D Q1 = P + V - d*N;
        AssertEq(Distance(Q1,line), d,
                 "Distance of P+d*N to line<P,V>"); }
}

void PointPlaneDistance()
{
    Point3D P(121.0f, -1290.2f, 347.04f);
    const float a=3.1f, b=4.2f, c=5.3f;
    const float d=6.1f, e=7.2f, f=8.3f;
    Vector3D U(a,b,c);
    Vector3D V(d,e,f);
    Vector3D N = normalize(cross(U,V));
    Plane3D plane(N[0], N[1], N[2], -dot(Vector3D(P[0],P[1],P[2]),N));

    AssertEq(Distance(P,plane), 0,
             "Distance of P to plane containing P");
    AssertEq(Distance(P+U+V,plane), 0,
             "Distance of P+U+V to plane containing P");

    for (float d=1;     d<=2;  d+=0.5) {
        Point3D Q0 = P + U + V + d*N;
        AssertEq(Distance(Q0,plane), d,
                 "Distance of point from plane.");
        Point3D Q1 = P + U - V - d*N;
        AssertEq(Distance(Q1,plane), d,
                 "Distance of point from plane."); }
}

void AngleBetweenPlanes()
{
    const float a=3.1f, b=4.2f, c=5.3f;
    const float d=6.1f, e=7.2f, f=8.3f;
    Plane3D P(a, b, c, 12.0);
    Plane3D Q(d, e, f, 13.0);

    float cos0 = cos(AngleBetween(P,Q));
    float cos1 = dot(normalize(P.normal()), normalize(Q.normal()));

    AssertEq(cos0, cos1, "Angle between planes.");
}

void AngleBetweenLinePlane()
{
    Vector3D V(-37.0f, 31.0f, -23.0f);
    Vector3D W(44.0f, -345.0f, 883.0f);
    Plane3D P(V[0], V[1], V[2], 1.0f);
    Line3D L(Point3D(255.45f, -42.78f, 363.0f), W);

    float cos0 = cos(AngleBetween(L,P));
    float cos1 = dot(V,W)/(length(V)*length(W));

    AssertEq(cos0*cos0 + cos1*cos1, 1.0f, "Angle between line and plane.");
}

void AngleBetweenLines()
{
    Vector3D V(-37.0f, 31.0f, -23.0f);
    Vector3D W(44.0f, -345.0f, 883.0f);
    Line3D M(Point3D(323.0f, 45.0f, -457.0f), V);
    Line3D L(Point3D(255.45f, -42.78f, 363.0f), W);

    float cos0 = cos(AngleBetween(M,L));
    float cos1 = dot(V,W);
    float l = length(V)*length(W);

    AssertEq(cos0, cos1/l, "Angle between lines.");
}

void CoplanarLines()
{
    Point3D Base(1.0f,2.0f,3.0f);
    Vector3D N(0.1f, 0.2f, 1.0f);
    Vector3D V (1.0f, 0.3f, 0.2f);
    Vector3D W=cross(N,V);

    if (Coplanar(Line3D(Base,V),Line3D(Base,W))) {
        AssertHere; }
    else {
        AssertNotHere("Coplanar lines with same base."); }

    if (Coplanar(Line3D(Base,V),Line3D(Base+V,W))) {
        AssertHere; }
    else {
        AssertNotHere("Coplanar lines with different base."); }

    if (Coplanar(Line3D(Base,V),Line3D(Base+N,W))) {
        AssertNotHere("NON-Coplanar lines with different base."); }
    else {
        AssertHere; }
}

void ParallelPerpendicular()
{
    Point3D P0(1,1,0);
    Point3D P1(1,0,1);
    Vector3D V(1,2,3);
    Vector3D W = cross(V, Vector3D(1,0,1));

    Line3D LW0 = Line3D(P0,W);
    Line3D LW1 = Line3D(P1,W);
    Line3D LV1 = Line3D(P1,V);

    Plane3D PW0(W[0], W[1], W[2], 0);
    Plane3D PW1(W[0], W[1], W[2], 1);
    Plane3D PV1(V[0], V[1], V[2], 1);

    AssertEq(true,  Parallel(LW0, LW1), "Parallel lines.");
    AssertEq(false, Parallel(LW0, LV1), "Not parallel lines.");
    AssertEq(false, Perpendicular(LW0, LW1), "Not perpendicular lines.");
    AssertEq(true,  Perpendicular(LW0, LV1), "Perpendicular lines.");

    AssertEq(true,  Parallel(PW0, PW1), "Parallel planes.");
    AssertEq(false, Parallel(PW0, PV1), "Not parallel planes.");
    AssertEq(false, Perpendicular(PW0, PW1), "Not perpendicular planes.");
    AssertEq(true,  Perpendicular(PW0, PV1), "Perpendicular planes.");

    AssertEq(false,  Parallel(LW0, PW1), "Not parallel line/plane.");
    AssertEq(true, Parallel(LW0, PV1), "Parallel line/plane.");
    AssertEq(true, Perpendicular(LW0, PW1), "Perpendicular line/plane.");
    AssertEq(false,  Perpendicular(LW0, PV1), "Not perpendicular line/plane.");
       
}


void Line2DIntersection()
{
    Line2D first(Point2D(15, 6), Vector2D(9, 3));
    Line2D second(Point2D(3, 0), Vector2D(0, 9));

    float t;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t), Point2D(3,2), "Line2D/Line2D intersection."); }
    else {
        AssertNotHere("Line2D/Line2D intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Line2D/Line2D intersection with NULL pointer."); }

    first.point = Point2D(0, 0);
    first.vector = Vector2D(9, 3);
    second.point = Point2D(3, 0);
    second.vector = Vector2D(9, 3);

    if (Intersects(first, second, &t)) {
        AssertNotHere("Line2D/Line2D non-intersection."); }
    else {
        AssertHere; }

}

void Segment2DIntersection()
{
    Segment2D first(Point2D(0, 0), Point2D(12, 12));
    Segment2D second(Point2D(0, 12), Point2D(12, 0));

    float t;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t), Point2D(6,6),
                 "Segment2D/Point2D intersection point."); }
    else {
        AssertNotHere("Segment2D/Point2D intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Segment2D/Point2D intersection with NULL pointer."); }

    first.point1 = Point2D(0, 0);
    first.point2 = Point2D(12, 12);
    second.point1 = Point2D(48, 48);
    second.point2 = Point2D(13, 13);

    if (Intersects(first, second)) {
        AssertNotHere("Segment2D/Point2D  non-intersection."); }
    else {
        AssertHere; }
}

void LinePlaneIntersection()
{
    Line3D first(Point3D(2.23f, 123.0f, 401.0f), Vector3D(123.0f, -13.0f, 63.3f));
    Plane3D second(32.0f, 3.5f, 21.0f, -12.0f);

    float t;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t),
                 Point3D(-2.0774658530978e+02f,
                         1.4519264722786e+02f,
                         2.9293887926740e+02f),
                 "Line3D/Plane3D intersection point."); }
    else {
        AssertNotHere("Line3D/Plane3D intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Line3D/Plane3D intersection with NULL pointer."); }

    first.point = Point3D(0.0f, 0.0f, 1.0f);
    first.vector = Vector3D(0.0f, 0.0f, 1.0f);
    second[0] = 1.0f;
    second[1] = 0.0f;
    second[2] = 0.0f;
    second[3] = 0.0f;

    if (Intersects(first, second)) {
        AssertNotHere("Line3D/Plane3D non-intersection."); }
    else {
        AssertHere; }
}

void SegmentPlaneIntersection()
{
    Point3D base(2.23f, 123.0f, 401.0f);
    Vector3D vec(-123.0f, -13.0f, 63.3f);

    Segment3D seg(base, base+vec);
    Plane3D plane(3.2f, 3.5f, .21f, -120.0f);

    float t;
    if (Intersects(seg, plane, &t))
    {
        Point3D point = seg.lerp(t);
        AssertEq(0.0, Distance(point,plane),
                 "Segment3D/Plane3D intersection point on plane.");
        AssertEq(0.0, Distance(point,Line3D(base,vec)),
                 "Segment3D/Plane3D intersection point on line.");
        AssertEq(true, seg.contains(point),
                 "Segment3D/Plane3D intersection point on Segment.");
    }
    else {
        AssertNotHere("Segment3D/Plane3D intersection."); }

    try {
        Intersects(seg, plane);
        AssertHere; }
    catch(...) {
        AssertNotHere("Segment3D/Plane3D intersection with NULL pointer."); }

    seg.point1 = Point3D(0, 0, 1);
    seg.point2 = seg.point1+Vector3D(0, 0, 1);
    plane[0] = 1;
    plane[1] = 0;
    plane[2] = 0;
    plane[3] = 0;

    if (Intersects(seg, plane)) {
        AssertNotHere("Segment3D/Plane3D non-intersection."); }
    else {
        AssertHere; }
}

void Ray3DPointContainment()
{ float t;

    {   Ray3D RAY(Point3D(), Vector3D(1.0f, 1.0f, 1.0f));
        Point3D P(-2.0f, -2.0f, -2.0f);
        if (RAY.contains(P, &t)) {
            AssertNotHere("Ray3D/Point3D non-containment"); }
        else {
            AssertHere; }
    }

    {   Ray3D RAY(Point3D(1.0f, 0.0f, 2.0f), Vector3D(4.0f, 8.0f, 6.0f));
        Point3D P(3.0f, 4.0f, 5.0f);

        if (RAY.contains(P, &t)) {
            AssertEq(t, 0.5f, "Ray3D/Point3D containment parameter value."); }
        else {
            AssertNotHere("Ray3D/Point3D containment."); }

        try {
            RAY.contains(P);
            AssertHere; }
        catch(...) {
            AssertNotHere("Ray3D/Point3D containment with NULL pointer."); }
    }

}

void Box3DPointContainment()
{
    Box3D BOX(Point3D(0.5f, 0.5f, 0.5f), Point3D(0.5f, 0.5f, 0.5f));
    
    Point3D points[] = {
        Point3D(),
        Point3D(1.0f,1.0f,1.0f),
        Point3D(0.1f, 0.1f, 0.1f)};

    for (int i=0;  i<sizeof(points)/sizeof(points[0]);  i++) {
        if (BOX.contains(points[i])) {
            AssertHere; }
        else {
            AssertNotHere("Box3D/Point3D containment."); }
        
        if (BOX.contains(points[i]+Vector3D(2.0f,0.0f,0.0f))) {
            AssertNotHere("Box3D/Point3D non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+Vector3D(-2.0f,0.0f,0.0f))) {
            AssertNotHere("Box3D/Point3D non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+Vector3D(0.0f,2.0f,0.0f))) {
            AssertNotHere("Box3D/Point3D non-containment"); }
        else {
            AssertHere; }
        
        if (BOX.contains(points[i]+Vector3D(0.0f,-2.0f,0.0f))) {
            AssertNotHere("Box3D/Point3D non-containment"); }
        else {
            AssertHere; } }
}

void Segment3DPointContainment()
{
    {   Segment3D seg(Point3D(1.0f, 0.0f, 2.0f), Point3D(5.0f, 8.0f, 8.0f));
        Point3D P(3.0f, 4.0f, 5.0f);

        if (seg.contains(P)) {
            AssertHere; }
        else {
            AssertNotHere("Segment3D/Point3D containment."); }
    }

    {   Segment3D seg(Point3D(), Point3D(1.0f, 1.0f, 1.0f));
        Point3D P(2.0f, 2.0f, 2.0f);
        if (seg.contains(P)) {
            AssertNotHere("Segment3D/Point3D non-containment"); }
        else {
            AssertHere; }
    }

    {   Segment3D seg(Point3D(), Point3D(0.0f, 0.0f, 1.0f));
        Point3D P(0.0f, 0.0f, 2.0f);
        if (seg.contains(P)) {
            AssertNotHere("Z-parallel-Segment3D/Point3D non-containment"); }
        else {
            AssertHere; }
    }
        
    {   Segment3D seg(Point3D(), Point3D(0.0f, 0.0f, 1.0f));
        Point3D P(0.0f, 0.0f, 0.5f);
        if (seg.contains(P)) {
            AssertHere; }
        else {
            AssertNotHere("Z-parallel-Segment3D/Point3D containment"); }
    }
}

void TrianglePointContainment()
{
    Point3D A(4.0f, -2.0f, 4.0f);
    Point3D B(0.0f, -2.0f, 2.0f);
    Point3D C(3.0f, -2.0f, -1.0f);
    Triangle3D TRI(A,B,C);
    Point3D In = A + 0.4f*(B-A) + 0.3f*(C-A);
    Point3D Out0 = A + 0.6f*(B-A) + 0.7f*(C-A);
    Point3D Out1 = A - 0.4f*(B-A) + 0.3f*(C-A);
    Point3D Out2 = A + 0.4f*(B-A) - 0.3f*(C-A);

    if (TRI.contains(In)) {
        AssertHere; }
    else {
        AssertNotHere("Triangle3D/Point3D containment."); }

    if (TRI.contains(Out0)) {
        AssertNotHere("Triangle3D/Point3D non-containment."); }
    else {
        AssertHere; }

    if (TRI.contains(Out1)) {
        AssertNotHere("Triangle3D/Point3D non-containment."); }
    else {
        AssertHere; }

    if (TRI.contains(Out2)) {
        AssertNotHere("Triangle3D/Point3D non-containment."); }
    else {
        AssertHere; }
}

void Segment3DTriangleIntersection()
{
    Segment3D first(Point3D(3, 0, 0), Point3D(3, 3, 6));
    Triangle3D second(Point3D(0, 0, 6), Point3D(6, 0, 6),
                      Point3D(3, 3, 0));
    float t;
    Point3D point;
    if (Intersects(first, second, &t)) {
        AssertEq(first.lerp(t), Point3D(3,1.5,3),
                 "Segment3D/Triangle3D intersection point."); }
    else {
        AssertNotHere("Segment3D/Triangle3D intersection."); }

    first.point1 = Point3D(10, 10, 10);
    first.point2 = Point3D(20, 20, 20);
    second[0] = Point3D(1, 0, 0);
    second[1] = Point3D(-1, 1, 0);
    second[2] = Point3D(0, -1, 0);

    if (Intersects(first, second)) {
        AssertNotHere("Segment3D/Triangle3D non-intersection"); }
    else {
        AssertHere; }
}

void TriangleIntersection()
{
    Triangle3D first(Point3D(0, 0, 6), Point3D(6, 0, 6),
                     Point3D(3, 3, 0));
    Triangle3D second(Point3D(0, 3, 4), Point3D(6, 3, 4),
                      Point3D(3.0f, 0.0f, 3.0f));

    std::pair <Point3D, Point3D> points;
    int intersection_count = Intersects(first, second, &points);

    if (intersection_count == 2)
    {
        if (points.first[0] > points.second[0]) {
            AssertEq(points.first,  Point3D(4.285714f, 1.285714f, 3.428571f),
                     "Triangle3D/Triangle3D intersection -- first point.");
            AssertEq(points.second, Point3D(1.714285f, 1.285714f, 3.428571f),
                     "Triangle3D/Triangle3D intersection -- second point."); }
        else {
            AssertEq(points.first,  Point3D(1.714285f, 1.285714f, 3.428571f),
                     "Triangle3D/Triangle3D intersection -- first point.");
            AssertEq(points.second, Point3D(4.285714f, 1.285714f, 3.428571f),
                     "Triangle3D/Triangle3D intersection -- second point."); } }
    else {
        AssertNotHere("Triangle3D/Triangle3D intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Triangle3D/Triangle3D intersection with NULL pointer.");}
}

void RaySphereIntersection()
{
    Ray3D first(Point3D(0, 4, 6), Vector3D(5, 3, -5));
    Sphere3D second(Point3D(2.5, 5.5, 3.5), 1);
    
    float t;
    int i = Intersects(first, second, &t);

    if (i) {
        AssertEq(length(first.lerp(t)-second.center),  second.radius,
                 "Ray3D/Sphere3D intersection."); }
    else {
        AssertNotHere("Ray3D/Sphere3D intersection."); }

    try {
        Intersects(first, second);
        AssertHere; }
    catch(...) {
        AssertNotHere("Ray3D/Sphere3D intersection with NULL pointer."); }
}

void TestRayBox(Point3D M, Vector3D V, int n, Point3D A, Point3D B)
{
    Ray3D ray(M,V);
    Box3D box(Point3D(0.5, 0.5, 0.5), Vector3D(0.5, 0.5, 0.5));
    float t;
    int i = Intersects(ray, box, &t);

    if (n==0)
        AssertEq(i, false, "Ray/Box non-intersection.");
    else {
        AssertEq(i, true, "Ray/Box intersection.");
        AssertEq(ray.lerp(t), A, "Ray/Box intersection."); }
}

void RayBoxIntersection()
{   
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               1, Point3D( 0.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               1, Point3D( 0.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               1, Point3D( 0.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 1.00, 0.50), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 1.00, 0.50), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 1.00, 0.50), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               1, Point3D( 0.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.50, 0.00),
               1, Point3D( 0.00, 0.75, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.50),
               1, Point3D( 0.00, 0.50, 0.75), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.50, 0.00),
               2, Point3D( 0.00, 0.75, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.50),
               2, Point3D( 0.00, 0.50, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 0.50, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.50, 0.00),
               2, Point3D( 0.00, 0.75, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.50),
               2, Point3D( 0.00, 0.50, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.00, 0.00),
               2, Point3D( 0.00, 1.00, 0.50), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.50, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 1.00, 0.50), Vector3D( 1.00, 0.00, 0.50),
               2, Point3D( 0.00, 1.00, 0.75), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.50, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D(-0.50, 1.50, 0.50), Vector3D( 1.00, 0.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               1, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               1, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               1, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 1.00, 0.50), Point3D( 0.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 1.00, 0.50), Point3D( 0.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 1.00, 0.50), Point3D( 0.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.00),
               1, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.50, 0.00),
               1, Point3D( 1.00, 0.75, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 1.00, 0.00, 0.50),
               1, Point3D( 1.00, 0.50, 0.75), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.50, 0.00),
               2, Point3D( 1.00, 0.75, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.50),
               2, Point3D( 1.00, 0.50, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 0.50, 0.50), Point3D( 0.00, 0.50, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.50, 0.00),
               2, Point3D( 1.00, 0.75, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 0.50), Vector3D(-1.00, 0.00, 0.50),
               2, Point3D( 1.00, 0.50, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.00, 0.00),
               2, Point3D( 1.00, 1.00, 0.50), Point3D( 0.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.50, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 1.00, 0.50), Vector3D(-1.00, 0.00, 0.50),
               2, Point3D( 1.00, 1.00, 0.75), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.50, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 1.50, 0.50), Vector3D(-1.00, 0.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00,-1.00),
               1, Point3D( 0.50, 0.50, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00,-1.00),
               1, Point3D( 0.50, 0.50, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00,-1.00),
               1, Point3D( 0.50, 0.50, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 1.00, 0.50, 0.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 1.00, 0.50, 0.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 1.00, 0.50, 0.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00,-1.00),
               1, Point3D( 0.50, 0.50, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.50, 0.00,-1.00),
               1, Point3D( 0.75, 0.50, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.50,-1.00),
               1, Point3D( 0.50, 0.75, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.50, 0.00, 1.00),
               2, Point3D( 0.75, 0.50, 0.00), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.50, 1.00),
               2, Point3D( 0.50, 0.75, 0.00), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 0.50, 0.50, 0.00), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.50, 0.00, 1.00),
               2, Point3D( 0.75, 0.50, 0.00), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 0.50,-0.50), Vector3D( 0.00, 0.50, 1.00),
               2, Point3D( 0.50, 0.75, 0.00), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               2, Point3D( 1.00, 0.50, 0.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.50, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.00, 0.50,-0.50), Vector3D( 0.00, 0.50, 1.00),
               2, Point3D( 1.00, 0.75, 0.00), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.00, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.50, 0.00, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50,-0.50), Vector3D( 0.00, 0.50, 1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00, 1.00),
               1, Point3D( 0.50, 0.50, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00, 1.00),
               1, Point3D( 0.50, 0.50, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00, 1.00),
               1, Point3D( 0.50, 0.50, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 1.00, 0.50, 1.00), Point3D( 1.00, 0.50, 0.00));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 1.00, 0.50, 1.00), Point3D( 1.00, 0.50, 0.00));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 1.00, 0.50, 1.00), Point3D( 1.00, 0.50, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.00, 1.00),
               1, Point3D( 0.50, 0.50, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.50, 0.00, 1.00),
               1, Point3D( 0.75, 0.50, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 0.50, 1.00),
               1, Point3D( 0.50, 0.75, 1.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.50, 0.00,-1.00),
               2, Point3D( 0.75, 0.50, 1.00), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.50,-1.00),
               2, Point3D( 0.50, 0.75, 1.00), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 0.50, 0.50, 1.00), Point3D( 0.50, 0.50, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.50, 0.00,-1.00),
               2, Point3D( 0.75, 0.50, 1.00), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 0.50, 1.50), Vector3D( 0.00, 0.50,-1.00),
               2, Point3D( 0.50, 0.75, 1.00), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               2, Point3D( 1.00, 0.50, 1.00), Point3D( 1.00, 0.50, 0.00));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.50, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.00, 0.50, 1.50), Vector3D( 0.00, 0.50,-1.00),
               2, Point3D( 1.00, 0.75, 1.00), Point3D( 1.00, 1.00, 0.50));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.00, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.50, 0.00,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 1.50, 0.50, 1.50), Vector3D( 0.00, 0.50,-1.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               1, Point3D( 0.50, 0.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               1, Point3D( 0.50, 0.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               1, Point3D( 0.50, 0.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 1.00), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 1.00), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 1.00), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.00, 1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.00, 1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.00, 1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               1, Point3D( 0.50, 0.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00,-1.00, 0.50),
               1, Point3D( 0.50, 0.00, 0.75), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.50,-1.00, 0.00),
               1, Point3D( 0.75, 0.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.50),
               2, Point3D( 0.50, 0.00, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.50, 1.00, 0.00),
               2, Point3D( 0.75, 0.00, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 0.50), Point3D( 0.50, 1.00, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.00, 1.00, 0.50),
               2, Point3D( 0.50, 0.00, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 0.50), Vector3D( 0.50, 1.00, 0.00),
               2, Point3D( 0.75, 0.00, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.00, 1.00, 0.00),
               2, Point3D( 0.50, 0.00, 1.00), Point3D( 0.50, 1.00, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.00, 1.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.00), Vector3D( 0.50, 1.00, 0.00),
               2, Point3D( 0.75, 0.00, 1.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.00, 1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.00, 1.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50,-0.50, 1.50), Vector3D( 0.50, 1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               1, Point3D( 0.50, 1.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               1, Point3D( 0.50, 1.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               1, Point3D( 0.50, 1.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 1.00), Point3D( 0.50, 0.00, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 1.00), Point3D( 0.50, 0.00, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 1.00), Point3D( 0.50, 0.00, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.00,-1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.00,-1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.00,-1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 1.00, 0.00),
               1, Point3D( 0.50, 1.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.00, 1.00, 0.50),
               1, Point3D( 0.50, 1.00, 0.75), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 0.50, 0.50), Vector3D( 0.50, 1.00, 0.00),
               1, Point3D( 0.75, 1.00, 0.50), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.50),
               2, Point3D( 0.50, 1.00, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.50,-1.00, 0.00),
               2, Point3D( 0.75, 1.00, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 0.50), Point3D( 0.50, 0.00, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.00,-1.00, 0.50),
               2, Point3D( 0.50, 1.00, 0.75), Point3D( 0.50, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 0.50), Vector3D( 0.50,-1.00, 0.00),
               2, Point3D( 0.75, 1.00, 0.50), Point3D( 1.00, 0.50, 0.50));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.00,-1.00, 0.00),
               2, Point3D( 0.50, 1.00, 1.00), Point3D( 0.50, 0.00, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.00,-1.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.00), Vector3D( 0.50,-1.00, 0.00),
               2, Point3D( 0.75, 1.00, 1.00), Point3D( 1.00, 0.50, 1.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.00,-1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.00,-1.00, 0.50),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
    TestRayBox(Point3D( 0.50, 1.50, 1.50), Vector3D( 0.50,-1.00, 0.00),
               0, Point3D( 0.00, 0.00, 0.00), Point3D( 0.00, 0.00, 0.00));
}


void RayTriangleIntersection()
{
    Point3D P0(0,0,0), P1;      // Points defining all rays
    Point3D A(1,0,0);           // Three points defining a triangle
    Point3D B(0,1,0);
    Point3D C(0,0,1);
    Triangle3D tri(A,B,C);
    Point3D O(0,0,0);

    // Generate three test
    for (int e=-1;  e<=1;  e++) {
        float a = e*0.1f;
        float b = (1.0f-a)/2.0f;
        float c = b;
        // for each of three edges
        for (int i=0;  i<3;  i++) {
            int j = (i+1) % 3;
            int k = (j+1) % 3;
            
            float t;
            
            
            P1 = O + a*(tri[i]-O) + b*(tri[j]-O) + c*(tri[k]-O);
            Ray3D ray(P0, P1-P0);
            bool retB = Intersects(ray, tri, &t);
            Point3D retP = ray.lerp(t);

            switch (e) {
            case -1:
                AssertEq(retB, false, "Ray3D/Triangle3D non-intersection.");
                break;
            case 0:
                AssertEq(retB, true, "Ray3D/Triangle3D edge intersection.");
                AssertEq(retP, P1,   "Ray3D/Triangle3D edge intersection.");
                break;
            case 1:
                AssertEq(retB, true, "Ray3D/Triangle3D interior intersection.");
                AssertEq(retP, P1,   "Ray3D/Triangle3D interior intersection.");
                break;
            }
        }
    }

    try {
        Intersects(Ray3D(P0, P1-P0), tri);
        AssertHere; }
    catch(...) {
        AssertNotHere("Ray3D/Triangle3D intersection with NULL pointer."); }

}

#define TEST(name, fn)                              \
    try { \
        fn(); \
    } catch(const Unimplemented&) { \
        if (!failures++) std::cout << "Failed tests:" <<  std::endl; \
        std::cout << "  " << name << " called unimplemented code." << std::endl; \
    }


int main()
{
    std::cout << std::endl;

    TEST("PointLineDistance", PointLineDistance);
    TEST("PointPlaneDistance", PointPlaneDistance);
    TEST("AngleBetweenPlanes", AngleBetweenPlanes);
    TEST("AngleBetweenLines", AngleBetweenLines);
    TEST("AngleBetweenLinePlane", AngleBetweenLinePlane);
    TEST("CoplanarLines", CoplanarLines);
    TEST("ParallelPerpendicular", ParallelPerpendicular);
    TEST("Line2DIntersection", Line2DIntersection);
    TEST("Segment2DIntersection", Segment2DIntersection);
    TEST("LinePlaneIntersection", LinePlaneIntersection);
    TEST("SegmentPlaneIntersection", SegmentPlaneIntersection);
    TEST("Ray3DPointContainment", Ray3DPointContainment);
    TEST("Box3DPointContainment", Box3DPointContainment);
    TEST("Segment3DPointContainment", Segment3DPointContainment);
    TEST("TrianglePointContainment", TrianglePointContainment);
    TEST("Segment3DTriangleIntersection", Segment3DTriangleIntersection);
    TEST("TriangleIntersection", TriangleIntersection);
    TEST("RaySphereIntersection", RaySphereIntersection);
    TEST("RayBoxIntersection", RayBoxIntersection);
    TEST("RayTriangleIntersection", RayTriangleIntersection);

    std::cout << std::endl;
    std::cout << completed << " tests completed." << std::endl;
    std::cout << failures << " tests failed." << std::endl;
        
    std::cout << "Press RETURN to quit: ";
    char x;
    scanf("%c", &x);
}
