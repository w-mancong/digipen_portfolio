/* Start Header *****************************************************************/
/*! \file (torus.tese)

     \author (Wong Man Cong, w.mancong, 390005621)

     \par (email: w.mancong\@digipen.edu)

     \date (April 4th, 2024)

     \brief Copyright (C) 2024 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */
/* End Header *******************************************************************/

#version 460

layout(quads, equal_spacing, cw) in;

layout (binding = 1) uniform UBOTessEval
{
  mat4 proj;
  mat4 view;
	vec4 center;
	float r;
	float R;
} ubo;

layout (location = 0) out vec3 tColor;

const float PI = 3.1415926535897932384626433832795;
const float TWO_PI = 2.0 * PI;

void main()
{
  const vec3 center = ubo.center.xyz;
  const float r = ubo.r, R = ubo.R, u = gl_TessCoord.x, v = gl_TessCoord.y;
  
  const float PHI = TWO_PI * u, THETA = TWO_PI * v, COS_PHI = cos(PHI);

  // Calculate torus vertex position using parametric equations
  vec3 torusPos = vec3(
    (R + r * COS_PHI) * cos(THETA),
    (R + r * COS_PHI) * sin(THETA),
    (r * sin(PHI))
  );
  
  // Calculate the color value for fragment shader to use
  tColor = clamp(torusPos, 0.0, 1.0);
  gl_Position = ubo.proj * ubo.view * vec4(center + torusPos, 1.0);
}