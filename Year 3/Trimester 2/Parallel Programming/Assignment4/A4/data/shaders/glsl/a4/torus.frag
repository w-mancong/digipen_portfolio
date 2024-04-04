/* Start Header *****************************************************************/
/*! \file (torus.frag)

     \author (Wong Man Cong, w.mancong, 390005621)

     \par (email: w.mancong\@digipen.edu)

     \date (April 4th, 2024)

     \brief Copyright (C) 2024 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */
/* End Header *******************************************************************/

#version 460

layout (location = 0) in vec3 tColor;

layout (location = 0) out vec4 fColor;

void main()
{
  fColor = vec4(tColor, 1.0);
}
