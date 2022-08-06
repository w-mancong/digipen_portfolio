/*!
@file	my-tutorial-5.frag
@author	w.mancong@digipen.edu
@date	16/06/2022

Fragment shader does a total of 3 different task:
Task 0: Rendering and rectangle filling up the entire screen and interpolates
the colour taken from the vertex shader

Task 1: 
Decide the color of the current fragment based on it's position

Task 2:
Same as task 1 but difference is that uniform uSize will be changed overtime
over on the client side

Task 3, 4, 5, 6:
Render texture onto the screen
*//*__________________________________________________________________________*/
#version 450 core

layout (location = 0) in vec3 vColor;
layout (location = 1) in vec2 vTexCoords;

layout (location = 0) out vec4 fColor;

uniform int uID, uModulate;
uniform float uSize;

vec4 clr1 = vec4(1.0, 0.0, 1.0, 1.0), clr2 = vec4(0.0, 0.68, 0.94, 1.0);

uniform sampler2D uTex2d;

void main()
{
	switch(uID)
	{
		case 0:
		{	
			fColor = vec4(vColor, 1.0);
			break;
		}
		case 1: case 2:
		{
			int tx = int( floor(gl_FragCoord.x / uSize) );
			int ty = int( floor(gl_FragCoord.y / uSize) );

			if(uModulate == 0)
				fColor = bool( mod(tx + ty, 2) ) ? clr1 : clr2;
			else
				fColor = (bool( mod(tx + ty, 2) ) ? clr1 : clr2) * vec4(vColor, 1.0);

			break;
		}
		case 3: case 4: case 5: case 6:
		{
			if(uModulate == 0)
				fColor = texture(uTex2d, vTexCoords);
			else 
				fColor = texture(uTex2d, vTexCoords) * vec4(vColor, 1.0);
			break;
		}
		default:			
			break;
	}
}