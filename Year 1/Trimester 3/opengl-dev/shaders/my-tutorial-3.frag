/*!
@file	my-tutorial-1.frag
@author	w.mancong@digipen.edu
@date	19/05/2022

Fragment shader that interpolate colors from each vertex and output it 
to the screen
*//*__________________________________________________________________________*/
#version 450 core

layout (location = 0) in vec3 vInterpColor;
layout (location = 0) out vec4 fColor;

void main()
{
	fColor = vec4(vInterpColor, 1.0f);
}