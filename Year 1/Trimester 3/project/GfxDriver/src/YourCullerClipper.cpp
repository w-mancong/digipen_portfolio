/*!
@file		YourCullerClipper.cpp
@author     Prasanna Ghali       (pghali@digipen.edu)
@co-author  Wong Man Cong        (w.mancong@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourCullerClipper.h"
#include <glm/glm.hpp>

/*                                                                  functions
----------------------------------------------------------------------------- */
/*  _________________________________________________________________________ */
/*! Get view frame frustum plane equations

  @param perspective_mtx	--> Matrix manifestation of perspective (or, orthographic)
  transform.

  @return	--> gfxFrustum
  Plane equations of the six surfaces that specify the view volume in view frame.
*/
gfxFrustum YourClipper::ComputeFrustum(gfxMatrix4 const&	perspective_mtx) 
{
	auto convert_to_glm = [](gfxMatrix4 const& perspective_mtx, unsigned int index)
	{
		gfxVector4 const& r = perspective_mtx.GetRow4(index);
		return glm::vec4{ r.x, r.y, r.z, r.w };
	};

	glm::vec4 const& r0 = convert_to_glm(perspective_mtx, 0), r1 = convert_to_glm(perspective_mtx, 1), r2 = convert_to_glm(perspective_mtx, 2), r3 = convert_to_glm(perspective_mtx, 3);

	gfxFrustum frustum;

	auto assign_plane = [](gfxPlane& plane, glm::vec4 const& val)
	{
		glm::vec3 l{ glm::normalize(val) };
		plane.a = l.x, plane.b = l.y, plane.c = l.z, plane.d = val.w;
	};

	// left plane
	assign_plane(frustum.l, -r0 - r3);

	// right plane
	assign_plane(frustum.r,  r0 - r3);

	// btm plane
	assign_plane(frustum.b, -r1 - r3);

	// top plane
	assign_plane(frustum.t,  r1 - r3);

	// near plane
	assign_plane(frustum.n, -r2 - r3);

	// far plane
	assign_plane(frustum.f,  r2 - r3);

	//@todo Implement me.
	return frustum;
}

/*  _________________________________________________________________________ */
/*! Performing culling.

@param bs		--> View-frame definition of the bounding sphere of object
which is being tested for inclusion, exclusion, or
intersection with view frustum.
@param f		--> View-frame frustum plane equations.
@param oc		--> Six-bit flag specifying the frustum planes intersected
by bounding sphere of object. A given bit of the outcode
is set if the sphere crosses the appropriate plane for that
outcode bit - otherwise the bit is cleared.

@return
True if the vertices bounded by the sphere should be culled.
False otherwise.

If the return value is false, the outcode oc indicates which planes
the sphere intersects with. A given bit of the outcode is set if the
sphere crosses the appropriate plane for that outcode bit.
*/
bool YourClipper::Cull(gfxSphere const& bs, gfxFrustum const& f, gfxOutCode *oc) 
{
	// reset culling information
	*oc = 0;

	glm::vec4 const c{ bs.center.x, bs.center.y, bs.center.z, 1.0f };
	float const r{ bs.radius };

	// checks if sphere on the outside region
	auto rejection_test = [=](gfxPlane const& plane)
	{
		glm::vec4 const l{ plane.a, plane.b, plane.c, plane.d };
		return glm::dot(l, c) > r;
	};

	// checks if sphere is intersecting with plane
	auto plane_intersection_test = [=](gfxPlane const& plane)
	{
		glm::vec4 const l{ plane.a, plane.b, plane.c, plane.d };
		// calculate the distance of center point to plane
		float const len = glm::dot(l, c);
		return -r < len&& len < r;
	};

	// checking against all 6 planes
	for (size_t i = 0; i < 6; ++i)
	{
		gfxPlane const& l = f.mPlanes[i];
		if (rejection_test(l))
			return true;
		if (plane_intersection_test(l))
			*oc |= (1 << i);
	}

	return false;
}

/*  _________________________________________________________________________ */
/*!
Perform clipping.

@param oc	--> Outcode of view-frame of bounding sphere of object specifying
the view-frame frustum planes that the sphere is straddling.

@param p	--> The input vertex buffer p contains three points forming a triangle.
Each vertex has x_c, y_c, z_c, and w_c fields that describe the position
of the vertex in clip space. Additionally, each vertex contains an array
of floats (bs[6]) that contains the boundary condition for each clip plane,
and an outcode value specifying which planes the vertex is inside.

The gfxClipPlane enum in GraphicsPipe.h contains the indices into the bs
array. gfxClipCode contains bit values for each clip plane code.

If an object's bounding sphere could not be trivially accepted nor rejected,
it is reasonable to expect that the object is straddling only a
subset of the six frustum planes. This means that the object's triangles
need not be clipped against all the six frustum planes but only against
the subset of planes that the bounding sphere is straddling.
Furthermore, even if the bounding sphere is straddling a subset of planes,
the triangles themselves can be trivially accepted or rejected.
To implement the above two insights, use argument oc - the object bounding
sphere's outcode which was previously returned by the Cull() function.

Notes:
When computing clip frame intersection points, ensure that all of the information
required to project and rasterize the vertex is computed using linear interpolation
between the inside and outside vertices. This includes:
clip frame coordinates: (c_x, c_y, c_z, c_w),
texture coordinates: (u, v),
vertex color coordinates: (r, g, b, a),
boundary conditions: bc[GFX_CPLEFT], bc[GFX_CPRIGHT], ...
outcode: oc

As explained in class, consistency in computing computing the parameter t using t = 0
for inside point and t = 1 for outside point helps in preventing tears and other artifacts.

Although the input primitive is a triangle, after clipping, the output
primitive may be a convex polygon with more than 3 vertices. In that
case, you must produce as output an ordered list of clipped vertices
that form triangles when taken in groups of three.

@return None
*/
gfxVertexBuffer YourClipper::Clip(gfxOutCode oc, const gfxVertexBuffer& vb)
{
	if (63 == (vb[0].oc & vb[1].oc & vb[2].oc)) return gfxVertexBuffer();
	if (0 == (vb[0].oc | vb[1].oc | vb[2].oc)) return vb;

	gfxVertexBuffer ob{ vb };
	using u32 = unsigned int;

	// p(t) = p0 + t(p1 - p0)
	auto parametric_eq = [](float p0, float p1, float t)
	{
		return p0 + t * (p1 - p0);
	};

	// using t, compute the intersection point with the plane
	auto intersection_point = [&](u32 plane, float t, gfxVertex const& p0, gfxVertex const& p1)
	{
		gfxVertex pi;

		// r, g, b, a
		pi.r = parametric_eq(p0.r, p1.r, t);
		pi.g = parametric_eq(p0.g, p1.g, t);
		pi.b = parametric_eq(p0.b, p1.b, t);
		pi.a = parametric_eq(p0.a, p1.a, t);

		// s, t
		pi.s = parametric_eq(p0.s, p1.s, t);
		pi.t = parametric_eq(p0.t, p1.t, t);

		// x, y, z, w
		pi.x_c = parametric_eq(p0.x_c, p1.x_c, t);
		pi.y_c = parametric_eq(p0.y_c, p1.y_c, t);
		pi.z_c = parametric_eq(p0.z_c, p1.z_c, t);
		pi.w_c = parametric_eq(p0.w_c, p1.w_c, t);

		// calculate boundary condition for pi
		for (size_t i = 0; i < 6; ++i)
		{
			pi.bc[i] = parametric_eq(p0.bc[i], p1.bc[i], t);
			pi.oc |= pi.bc[i] <= 0.0f ? 0 : (1 << i);
		}

		return pi;
	};

	// check against all 6 planes
	for (u32 i = 0; i < 6; ++i)
	{
		gfxVertexBuffer nb;
		// with all the vertices stored
		for (u32 j = 0; j < ob.size(); ++j)
		{
			gfxVertex const p0{ ob[j] }, p1{ ob[ (j + 1) % ob.size() ] };
			float const bc0{ p0.bc[i] }, bc1{ p1.bc[i] };
			// calculate t to find intersection point
			float const t = bc0 / (bc0 - bc1);

			// inside: bc <= 0, outside: bc > 0
			// Points go from p0 -> p1
			// Edge Processing Rule 1: outside -> inside (store pi and p0)
			if (bc0 > 0.0f && bc1 <= 0.0f)
			{
				gfxVertex pi = intersection_point(i, t, p0, p1);
				nb.emplace_back(pi); nb.emplace_back(p1);
			}

			// Edge Processing Rule 2: inside -> inside (store p1)
			else if (bc0 <= 0.0f && bc1 <= 0.0f)
				nb.emplace_back(p1);

			// Edge Processing Rule 3: inside -> outside (store pi)
			else if (bc0 <= 0.0f && bc1 > 0.0f)
			{
				gfxVertex pi = intersection_point(i, t, p0, p1);
				nb.emplace_back(pi);
			}
			// Edge Processing Rule 4: outside -> outside (reject)
		}
		ob = std::move(nb);
	}

	// create ordered list of clipped vertices that form triangles when taken in groups of three.
	if (!ob.size() || ob.size() == 3)
		return ob;

	gfxVertexBuffer nb;

	size_t const LAST_INDEX{ ob.size() - 2 }; size_t i{ 0 };
	while (i++ < LAST_INDEX)
	{
		for (size_t j = 0; j < 3; ++j)
			nb.emplace_back(ob[j]);

		// erase the 2nd index
		ob.erase(ob.begin() + 1);
	}

	// return the triangles
	return nb;
}
