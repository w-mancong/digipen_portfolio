/* Start Header *****************************************************************/
/*! \file (torus.tesc)

     \author (Wong Man Cong, w.mancong, 390005621)

     \par (email: w.mancong\@digipen.edu)

     \date (April 4th, 2024)

     \brief Copyright (C) 2024 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */
/* End Header *******************************************************************/

#version 460

layout (binding = 0) uniform UBOTessControl
{
  float tessLevel;
} ubo;
 
layout (vertices = 1) out;

void main()
{
  gl_TessLevelOuter[0] = ubo.tessLevel;
	gl_TessLevelOuter[1] = ubo.tessLevel;
	gl_TessLevelOuter[2] = ubo.tessLevel;
	gl_TessLevelOuter[3] = ubo.tessLevel;

	gl_TessLevelInner[0] = ubo.tessLevel;
	gl_TessLevelInner[1] = ubo.tessLevel;
	
  // Assign all invocations to be first value
	gl_out[gl_InvocationID].gl_Position = gl_in[0].gl_Position;
} 
