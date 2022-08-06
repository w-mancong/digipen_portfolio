/*!
@file	my-tutorial-5.vert
@author	w.mancong@digipen.edu
@date	16/06/2022

takes in data of each vertex attribute and transfer the data over to the
fragment shader. 
*//*__________________________________________________________________________*/
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoords;

// sending data over to fragment shader
layout (location = 0) out vec3 vColor;
layout (location = 1) out vec2 vTexCoords;

void main()
{
	gl_Position = vec4(aPos, 0.0f, 1.0f);
	vColor = aColor;
	vTexCoords = aTexCoords;
}