/*!
@file	my-tutorial-1.frag
@author	w.mancong@digipen.edu
@date	09/06/2022

Fragment shader that uses a uniform color to determine the color of the object
*//*__________________________________________________________________________*/
#version 450 core

layout (location = 0) out vec4 fColor;

uniform vec3 uColor;

void main()
{
	fColor = vec4(uColor, 1.0f);
}