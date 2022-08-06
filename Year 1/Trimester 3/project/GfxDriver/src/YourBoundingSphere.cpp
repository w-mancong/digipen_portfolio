/*!
@file       BoundingSphere.cpp
@author     Prasanna Ghali       (pghali@digipen.edu)
@co-author  Wong Man Cong        (w.mancong@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "gfx/GFX.h"
#include "glm/glm.hpp"

/*  _________________________________________________________________________ */
/*! Compute model's bounding sphere (in model frame) using Ritter's method.

    @param verts -->  Model-frame vertices of scene object. This function will be
                      called once for each model as it is loaded. You should compute
                      the bounding sphere containing all the input vertices
                      stored in the verts vector using Ritter's method.
                      See GfxLib/Model.h, GfxLib/Sphere.h, GfxLib/Vertex.h for more
                      information regarding gfxModel, gfxSphere, and gfxVertex class
                      declarations.

    @return
    Object of type gfxSphere defining the bounding sphere containing the model using
    Ritter's method.
*/
gfxSphere gfxModel::ComputeModelBVSphere(std::vector<gfxVector3> const& verts)
{
    glm::vec3 cen(0.0f);
    float rad{ 0.0f }, sq_rad{ 0.0f };

    // Pass 1: Calculating the largest pair and build a bounding sphere from there
    float constexpr MAX_FLOAT = 3.402823466e+38F, MIN_FLOAT = 1.175494351e-38F;

    struct Pair
    {
        // GIVE MIN AND MAX INITIAL VALUES
        glm::vec3 min{ MAX_FLOAT, MAX_FLOAT, MAX_FLOAT }, max{ MIN_FLOAT, MIN_FLOAT, MIN_FLOAT };
        float len{ 0.0f };
    };

    size_t constexpr X{ 0 }, Y{ 1 }, Z{ 2 };
    // max_pair: the pair with the largest magnitude
    Pair pair[3], max_pair;

    auto assign_value = [](glm::vec3& coord, gfxVector3 const& vex)
    {
        coord.x = vex.x, coord.y = vex.y, coord.z = vex.z;
    };

    // 1st Pass: Finding 6 maximum/minimum points
    for (size_t i = 0; i < verts.size(); ++i)
    {
        gfxVector3 const& vex = verts[i];

        if (vex.x < pair[X].min.x)
            assign_value(pair[X].min, vex);
        if (vex.x > pair[X].max.x)
            assign_value(pair[X].max, vex);

        if (vex.y < pair[Y].min.y)
            assign_value(pair[Y].min, vex);
        if (vex.y > pair[Y].max.y)
            assign_value(pair[Y].max, vex);

        if (vex.z < pair[Z].min.z)
            assign_value(pair[Z].min, vex);
        if (vex.z > pair[Z].max.z)
            assign_value(pair[Z].max, vex);
    }

    // Calculate the square radius
    glm::vec3 w = pair[X].max - pair[X].min;
    pair[X].len = glm::dot(w, w);

    w = pair[Y].max - pair[Y].min;
    pair[Y].len = glm::dot(w, w);

    w = pair[Z].max - pair[Z].min;
    pair[Z].len = glm::dot(w, w);

    // assume X has the greatest length
    max_pair = pair[X];
    if (pair[Y].len > max_pair.len)
        max_pair = pair[Y];
    if (pair[Z].len > max_pair.len)
        max_pair = pair[Z];

    // Calculate the centre point, radius and squared radius (used for Pass 2)
    sq_rad = max_pair.len * 0.25f;
    rad = glm::sqrt(sq_rad);
    cen = (max_pair.max + max_pair.min) * 0.5f;

    // Pass 2: Include any points that might be outside of the bounding sphere
    for (size_t i = 0; i < verts.size(); ++i)
    {
        glm::vec3 const& p{ verts[i].x, verts[i].y, verts[i].z };
        glm::vec3 u{ p - cen };
        float const sq_len = glm::dot(u, u);
        if (sq_len < sq_rad)
            continue;
        /*
            point p is outside of the sphere, compute new sphere
            Formula:
            u  = (p - c) / || (p - c) ||
            p' =  c - ru
            c' = (p' + p) / 2
            r' = || p - c' ||
        */ 
        u = glm::normalize(u);
        cen = ((cen - rad * u) + p) * 0.5f;
        rad = glm::length(p - cen);

        sq_rad = sq_len;
    }

    return (gfxSphere(cen.x, cen.y, cen.z, rad));
}

/*  _________________________________________________________________________ */
gfxSphere gfxSphere::Transform(gfxMatrix4 const& xform) const
/*! Transform a bounding sphere to a destination reference frame using
    the matrix manifestation of the transform from model to destination
    frame. Note that the only valid affine transforms for this assignment are:
    scale (if any), followed by rotation (if any), followed by translation.

    @param xform -->  The matrix manifestation of transform from model to
                      destination frame.

    @return
    The model bounding sphere transformed to destination frame.
*/
{
    gfxVector4 const cen{ xform * gfxVector4{ center.x, center.y, center.z, 1.0f } };
    float const len[3] = { xform.GetCol4(0).Length(), xform.GetCol4(1).Length(), xform.GetCol4(2).Length() };
    float const rad = radius * std::max( len[0], std::max( len[1], len[2] ) );
   
    //@todo Implement me.
    return (gfxSphere(cen.x, cen.y, cen.z, rad));
}
