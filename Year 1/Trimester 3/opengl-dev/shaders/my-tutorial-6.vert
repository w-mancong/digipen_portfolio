/*!
@file	my-tutorial-6.vert
@author	w.mancong@digipen.edu
@date	26/06/2022
*//*__________________________________________________________________________*/
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 2) in vec2 aTexCoords;

// sending data over to fragment shader
layout (location = 0) out vec2 vTexCoords;

void main()
{
	gl_Position = vec4(aPos, 0.0f, 1.0f);
	vTexCoords = aTexCoords;
}