/*!
@file		Camera.cpp
@author     Prasanna Ghali       (pghali@digipen.edu)
@co-author  Wong Man Cong        (w.mancong@digipen.edu)

CVS: $Id: Camera.cpp,v 1.13 2005/03/15 23:34:41 pghali Exp $

All content (c) 2005 DigiPen (USA) Corporation, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourCamera.h"

/*                                                                  functions
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
/*! Displace the camera along its basis vectors.

	@param p -->  Displacement vector such that p.x, p.y, and p.z are
				  displacements along the camera's side, up, and viewing
				  basis vectors.
*/
void YourCamera::Move(gfxVector3 const& p)
{
	Move(p.x, p.y, p.z);
}

/*  _________________________________________________________________________ */
/*! Displace the camera along its basis vectors.

	@param x -->  Displacement along camera's side (X) axis.
	@param y -->  Displacement along camera's up (Y) axis.
	@param z -->  Displacement along camera's view (-Z) axis.
*/
void YourCamera::Move(float x, float y, float z)
{
	// view vector
	gfxVector3 const view{ (mAt - mFrom).Norm() }, right{ (mUp ^ view).Norm() }, forward{ (right ^ mUp).Norm() };

	mFrom += x * right, mFrom += y * mUp, mFrom += z * forward;
	mAt   += x * right, mAt   += y * mUp, mAt   += z * forward;
}

/*  _________________________________________________________________________ */
/*! Updates camera's spherical coordinates using updated
		camera position mFrom and camera target mAt.
*/
void YourCamera::UpdateSphericalFromPoints()
{
	gfxVector3 const v{ mAt - mFrom };
	mRadius = v.Length(), mLatitude = std::asinf(v.y / mRadius), mAzimuth = 0.0f;
	float const x = v.x, z = v.z;

	if (x >= 0.0f && z != 0.0f)
		mAzimuth = std::atan2f(x, z);
	else if (x > 0.0f && z == 0.0f)
		mAzimuth = HALF_PI;
	else if (x < 0.0f && z != 0.0f)
		mAzimuth = std::atan2f(x, z) + TWO_PI;
	else if (x < 0.0f && z == 0.0f)
		mAzimuth = 3.0f * PI * 0.5f;
}

/*  _________________________________________________________________________ */
/*! Updates camera's target position mAt using camera's
		updated spherical coordinates.
*/
void YourCamera::UpdatePointsFromSpherical()
{
	float const cos_lat = std::cosf(mLatitude), sin_lat = std::sinf(mLatitude);
	float const cos_azi = std::cosf(mAzimuth),  sin_azi = std::sinf(mAzimuth);

	mAt.x = mFrom.x + mRadius * cos_lat * sin_azi;
	mAt.y = mFrom.y + mRadius * sin_lat;
	mAt.z = mFrom.z + mRadius * cos_lat * cos_azi;
}
