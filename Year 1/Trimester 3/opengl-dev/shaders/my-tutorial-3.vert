/*!
@file	my-tutorial-1.vert
@author	w.mancong@digipen.edu
@date	02/06/2022

takes in data of each vertex attribute and transfer the data over to the
fragment shader. Does multiplication with model matrix to transform the points
*//*__________________________________________________________________________*/
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 0) out vec3 vColor;

uniform mat3 uModel_to_NDC;

void main()
{
	gl_Position = vec4( vec2(uModel_to_NDC * vec3(aPos, 1.0) ), 0.0, 1.0 );
	vColor		= aColor;
}