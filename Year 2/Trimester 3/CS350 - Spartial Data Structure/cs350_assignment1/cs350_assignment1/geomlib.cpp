///////////////////////////////////////////////////////////////////////
// Geometric objects (Points, Vectors, Planes, ...) and operations.
//
////////////////////////////////////////////////////////////////////////

#pragma warning(disable:4996)

#include "geomlib.h"
#include <vector>

float ToRadians(const float& Degrees)
{
    const float ret = (PI*Degrees)/180.0f;
    return ret;
}

// Test any STL sequence for all elements zero
template<class Sequence> bool IsZero(const Sequence& s)
{
    for(typename Sequence::const_iterator i = s.begin(); i != s.end(); ++i)
        if(0 != *i)
            return false;
    return true;
}


Color HSVColor(float h, float s, float v)
{
    if (s == 0.0)
        return Color(v,v,v);
    else {
        int i = (int)(h*6.0);
        float f = (h*6.0f) - i;
        float p = v*(1.0f - s);
        float q = v*(1.0f - s*f);
        float t = v*(1.0f - s*(1.0f-f));
        if (i%6 == 0)
            return Color(v,t,p);
        else if (i == 1)
            return Color(q,v,p);
        else if (i == 2)
            return Color(p,v,t);
        else if (i == 3)
            return Color(p,q,v);
        else if (i == 4)
            return Color(t,p,v);
        else //if (i == 5)
            return Color(v,p,q); }
}

bool Coplanar(const Point3D& A,const Point3D& B,
              const Point3D& C, const Point3D& D)
{
    // Generated with maxima as the determinant of the 4x4:
    // | A[0] A[1] A[2] 1 |
    // |                  |
    // | B[0] B[1] B[2] 1 |
    // |                  |
    // | C[0] C[1] C[2] 1 |
    // |                  |
    // | D[0] D[1] D[2] 1 |
	float det = 
		- B[0]*C[1]*D[2] + A[0]*C[1]*D[2] + B[1]*C[0]*D[2] - A[1]*C[0]*D[2] 
		- A[0]*B[1]*D[2] + A[1]*B[0]*D[2] + B[0]*C[2]*D[1] - A[0]*C[2]*D[1] 
		- B[2]*C[0]*D[1] + A[2]*C[0]*D[1] + A[0]*B[2]*D[1] - A[2]*B[0]*D[1] 
		- B[1]*C[2]*D[0] + A[1]*C[2]*D[0] + B[2]*C[1]*D[0] - A[2]*C[1]*D[0] 
		- A[1]*B[2]*D[0] + A[2]*B[1]*D[0] + A[0]*B[1]*C[2] - A[1]*B[0]*C[2] 
		- A[0]*B[2]*C[1] + A[2]*B[0]*C[1] + A[1]*B[2]*C[0] - A[2]*B[1]*C[0];

	return fabs(det) < 1.0e-3;
}
