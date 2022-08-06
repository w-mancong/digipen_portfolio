/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		16/06/2022

This file implements methods to that toggles between drawing different things
onto the screen.

T: Toggle between different task
0: Colour painting, 1: Fixed Checkerboard, 2: Animated Checkerboard
3: Texture mapping, 4: Repeating textures, 5: Mirroring textures
6: Clamping to edge

M: Toggle modulation
decides if the current fragment should be interpolated with color of the vertex

A: Toggle between alpha blend mode
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#define _USE_MATH_DEFINES
#include <math.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
double fps						= 0.0;

GLApp::GLModel GLApp::rectModel;
GLSLShader GLApp::shdrpgm;
GLuint GLApp::duck_texture = 0;

int const MAX_MODE = 7;
int id = 0, modulate = 0, alpha = 0;

bool t_key_pressed = false, m_key_pressed = false, a_key_pressed = false;

float const TOTAL_TIME = 30.0f, MIN_SIZE = 16, MAX_SIZE = 256, NORMAL_SIZE = 32;
float curr_time = 0.0f, time_flag = 1.0f, uSize = NORMAL_SIZE;

GLenum s_mode = GL_REPEAT, t_mode = GL_REPEAT;

/*  _________________________________________________________________________ */
/*! ToogleState

@param	state: param shld be based on alpha and modulate

@return none

Helper function used to determine if alpha and modulate is toggled
Used to change the text on the window bar
*/
const char* ToogleState(int state)
{
	return state ? "ON" : "OFF";
}

/*  _________________________________________________________________________ */
/*! init

@param none

@return none

Initializes viewport, vao and shader program
*/
void GLApp::init() 
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	glViewport(0, 0, GLHelper::width, GLHelper::height);

	init_shader("../shaders/my-tutorial-5.vert", "../shaders/my-tutorial-5.frag");
	init_rect_model();

	shdrpgm.Use();
	shdrpgm.SetUniform("uSize", uSize);
	shdrpgm.UnUse();

	duck_texture = setup_texobj("../images/duck-rgba-256.tex");

	t_key_pressed = false, m_key_pressed = false, a_key_pressed = false;
	curr_time = 0.0f, time_flag = 1.0f;

	id = 0, modulate = 0, alpha = 0;

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

/*  _________________________________________________________________________ */
/*! update

@param none

@return none

Update inputs from user and if it's task 2, update the interpolation of the size
*/
void GLApp::update() 
{
	update_inputs();

	switch (id)
	{
		case 2:
		{
			update_time();
			shdrpgm.Use();
			shdrpgm.SetUniform("uSize", uSize);
			shdrpgm.UnUse();
			break;
		}
	}
}

/*  _________________________________________________________________________ */
/*! draw

@param none

@return none

clear current framebuffer and call the draw call to draw to previous framebuffer
*/
void GLApp::draw() 
{
	glClear(GL_COLOR_BUFFER_BIT);

	std::ostringstream oss;
	static double time = 1.0;
	time += GLHelper::delta_time;

	if (1.0 < time)
	{
		fps = GLHelper::delta_time < 0.0001 ? 0.0 : 1.0 / GLHelper::delta_time;
		time = 0.0;
	}

	std::string task;
	switch (id)
	{
		case 0:
		{
			task = "Task 0: Paint Colour";
			break;
		}
		case 1:
		{
			task = "Task 1: Fixed-Size Checkerboard";
			break;
		}
		case 2:
		{
			task = "Task 2: Animated Checkerboard";
			break;
		}
		case 3:
		{
			task = "Task 3: Texture Mapping";
			break;
		}
		case 4:
		{
			task = "Task 4: Repeating";
			break;
		}
		case 5:
		{
			task = "Task 5: Mirroring";
			break;
		}
		case 6:
		{
			task = "Task 6: Clamping";
			break;
		}
	}

	oss << std::fixed << std::setprecision(2) << "Tutorial 5 | Wong Man Cong | " << task <<
		" | Alpha Blend: " << ToogleState(alpha) <<
		" | Modulate: " << ToogleState(modulate) <<
		" | FPS: " << fps;

	glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());

	draw_rect_model();
}

/*  _________________________________________________________________________ */
/*! cleanup

@param none

@return none

empty for now
*/
void GLApp::cleanup() 
{
  // empty for now
}

/*  _________________________________________________________________________ */
/*! update_inputs

@param	none

@return none

check the inputs by the user and update the keys accordingly
The type of keys and it's functionalities are explained at the top
*/
void GLApp::update_inputs()
{
	/*******************************************************************************
							T KEY - Toggle between different modes
	*******************************************************************************/
	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_T) && !t_key_pressed)
	{
		(++id) %= MAX_MODE;
		shdrpgm.Use();
		shdrpgm.SetUniform("uID", id);

		switch (id)
		{
			case 1:
			{
				uSize = NORMAL_SIZE;
				shdrpgm.SetUniform("uSize", uSize);
				break;
			}
			case 2:
			{
				uSize = MIN_SIZE, time_flag = 1.0f, curr_time = 0.0f;
				shdrpgm.SetUniform("uSize", uSize);
				break;
			}
			case 3:
			{
				float vertex[] =
				{
					// Position,		  Color,		 Texture
					-1.0f, -1.0f,	1.0f, 0.0f, 0.0f,	0.0f, 0.0f,		// btm left
					 1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	1.0f, 0.0f,		// btm right
					 1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	1.0f, 1.0f,		// top right
					-1.0f,  1.0f,	0.3f, 0.3f, 0.3f,	0.0f, 1.0f,		// top left
				};

				s_mode = GL_REPEAT, t_mode = GL_REPEAT;

				glNamedBufferSubData(rectModel.vbo, 0, sizeof(vertex), vertex);
				break;
			}
			case 4:
			{
				float vertex[] =
				{
					// Position,		  Color,		 Texture
					-1.0f, -1.0f,	1.0f, 0.0f, 0.0f,	0.0f, 0.0f,		// btm left
					 1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	4.0f, 0.0f,		// btm right
					 1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	4.0f, 4.0f,		// top right
					-1.0f,  1.0f,	0.3f, 0.3f, 0.3f,	0.0f, 4.0f,		// top left
				};

				glNamedBufferSubData(rectModel.vbo, 0, sizeof(vertex), vertex);
				break;
			}
			case 5:
			{
				s_mode = GL_MIRRORED_REPEAT, t_mode = GL_MIRRORED_REPEAT;
				break;
			}
			case 6:
			{
				s_mode = GL_CLAMP_TO_EDGE, t_mode = GL_CLAMP_TO_EDGE;
				break;
			}
		}
		shdrpgm.UnUse();

		t_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_T) && t_key_pressed)
		t_key_pressed = false;

	/*******************************************************************************
							M KEY - Toggle Modulation
	*******************************************************************************/
	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_M) && !m_key_pressed)
	{
		(++modulate) %= 2;
		shdrpgm.Use();
		shdrpgm.SetUniform("uModulate", modulate);
		shdrpgm.UnUse();
		m_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_M) && m_key_pressed)
		m_key_pressed = false;

	/*******************************************************************************
							A KEY - Toggle Alpha
	*******************************************************************************/
	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_A) && !a_key_pressed)
	{
		(++alpha) %= 2;
		if (alpha)
			glEnable(GL_BLEND);
		else
			glDisable(GL_BLEND);

		a_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_A) && a_key_pressed)
		a_key_pressed = false;
}

/*  _________________________________________________________________________ */
/*! update_time

@param	none

@return none

linearly interpolate uSize between MIN_SIZE and MAX_SIZE over a period of TOTAL_TIME
*/
void GLApp::update_time()
{
	curr_time += static_cast<float>(GLHelper::delta_time) * time_flag;
	if (30.0f <= curr_time)
		time_flag = -1.0f;
	else if (0.0f >= curr_time)
		time_flag = 1.0f;

	float const t = curr_time / TOTAL_TIME;
	float const theta = static_cast<float>(M_PI) * t - static_cast<float>(M_PI_2);
	float const e = (std::sinf(theta) + 1.0f) * 0.5f;

	uSize = MIN_SIZE + e * (MAX_SIZE - MIN_SIZE);
}

/*  _________________________________________________________________________ */
/*! init_shader

@param	vtx_shdr: file path to vertex shader
		fgm_shdr: file path to fragment shader

@return none

initialise the shader program that gonna be used
*/
void GLApp::init_shader(std::string const& vtx_shdr, std::string const& fgm_shdr)
{
	std::vector<std::pair<GLenum, std::string>> shdr_files
	{
		std::make_pair(GL_VERTEX_SHADER, vtx_shdr),
		std::make_pair(GL_FRAGMENT_SHADER, fgm_shdr)
	};
	shdrpgm.CompileLinkValidate(shdr_files);
	if (GL_FALSE == shdrpgm.IsLinked())
	{
		std::cerr << "Unable to compile/link/validate shader programs" << std::endl;
		std::cerr << shdrpgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

/*  _________________________________________________________________________ */
/*!	init_rect_model

@param	none

@return none

initializes a model of a rectangle that covers the entire viewport using the 
Array Of Structures format (AOS)
*/
void GLApp::init_rect_model()
{
	// btm left, btm right, top right, top left
	float vertex[] = 
	{
		// Position,		  Color,		 Texture
		-1.0f, -1.0f,	1.0f, 0.0f, 0.0f,	0.0f, 0.0f,		// btm left
		 1.0f, -1.0f,	0.0f, 1.0f, 0.0f,	1.0f, 0.0f,		// btm right
		 1.0f,  1.0f,	0.0f, 0.0f, 1.0f,	1.0f, 1.0f,		// top right
		-1.0f,  1.0f,	0.3f, 0.3f, 0.3f,	0.0f, 1.0f,		// top left
	};

	GLuint vao = 0, vbo = 0, ebo = 0;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(vertex), vertex, GL_DYNAMIC_STORAGE_BIT);

	glCreateVertexArrays(1, &vao);
	// position attribute
	glEnableVertexArrayAttrib(vao, 0);
	glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(float) * 7);
	glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(vao, 1);
	glVertexArrayVertexBuffer(vao, 1, vbo, sizeof(float) * 2, sizeof(float) * 7);
	glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 1, 1);

	// texture attribute
	glEnableVertexArrayAttrib(vao, 2);
	glVertexArrayVertexBuffer(vao, 2, vbo, sizeof(float) * 5, sizeof(float) * 7);
	glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 2, 2);

	// ebo
	std::vector<GLushort> idx_vtx{ 1, 2, 0, 3 };
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLushort) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vao, ebo);
	
	// unbind vao to prevent unwanted changes
	glBindVertexArray(0);

	rectModel = GLModel{ GL_TRIANGLE_STRIP, vao, vbo, idx_vtx.size() };
}

/*  _________________________________________________________________________ */
/*! draw_rect_model

@param	none

@return none

draws the rectangle model
*/
void GLApp::draw_rect_model()
{
	glBindTextureUnit(0, duck_texture);
	glTextureParameteri(duck_texture, GL_TEXTURE_WRAP_S, s_mode);
	glTextureParameteri(duck_texture, GL_TEXTURE_WRAP_T, t_mode);

	shdrpgm.Use();

	GLuint tex_loc = glGetUniformLocation(shdrpgm.GetHandle(), "uTex2d");
	glUniform1i(tex_loc, 0);

	glBindVertexArray(rectModel.vaoid);
	glDrawElements(rectModel.primitive_type, rectModel.draw_cnt, GL_UNSIGNED_SHORT, nullptr);
	glBindVertexArray(0);

	shdrpgm.UnUse();
}

/*  _________________________________________________________________________ */
/*! setup_texobj

@param	file_path: path to the texture file

@return a handle to the texture object created

Read and create a texture object handle
*/
GLuint GLApp::setup_texobj(std::string const& file_path)
{
	GLuint constexpr width{ 256 }, height{ 256 };
	std::ifstream ifs{ file_path, std::ios::binary };
	ifs.seekg(0, ifs.end); size_t const LENGTH = static_cast<size_t>(ifs.tellg()); ifs.seekg(0, ifs.beg);

	char* ptr_texels = new char[LENGTH];
	ifs.read(ptr_texels, LENGTH);

	GLuint texobj_hdl{ 0 };
	glCreateTextures(GL_TEXTURE_2D, 1, &texobj_hdl);
	glTextureStorage2D(texobj_hdl, 1, GL_RGBA8, width, height);
	glTextureSubImage2D(texobj_hdl, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, ptr_texels);

	delete[] ptr_texels;

	return texobj_hdl;
}