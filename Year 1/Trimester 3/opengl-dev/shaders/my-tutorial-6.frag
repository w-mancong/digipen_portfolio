/*!
@file	my-tutorial-6.frag
@author	w.mancong@digipen.edu
@date	26/06/2022
*//*__________________________________________________________________________*/
#version 450 core

layout (location = 0) in vec2 vTexCoords;

layout (location = 0) out vec4 fColor;

uniform sampler2D uTex2d;

void main()
{
	fColor = texture(uTex2d, vTexCoords);
}