/*!
@file	my-tutorial-1.vert
@author	w.mancong@digipen.edu
@date	19/05/2022

takes in data of each vertex attribute and transfer the data over to the
fragment shader
*//*__________________________________________________________________________*/
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

layout (location = 0) out vec3 vColor;

void main()
{
	gl_Position = vec4(aPos, 0.0f, 1.0f);
	vColor		= aColor;
}