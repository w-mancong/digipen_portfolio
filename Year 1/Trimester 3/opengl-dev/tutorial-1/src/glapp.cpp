/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		19/05/2022

This file implements functionality useful and necessary to build OpenGL
applications including use of external APIs such as GLFW to create a
window and start up an OpenGL context and to extract function pointers
to OpenGL implementations.

*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
GLApp::GLModel GLApp::mdl;

/*  _________________________________________________________________________ */
/*! init

@param none

@return none

Initializes viewport, vao and shader program
*/
void GLApp::init() 
{
	// Part 1: clear colorbuffer with RGBA value with glClearColor
	glClearColor(0.0f, 1.0f, 0.0f, 1.0f);

	// Part 2: use entire window as viewport
	glViewport(0, 0, GLHelper::width, GLHelper::height);

	// Part 3: initialize VAO and create shader program
	mdl.setup_vao();
	mdl.setup_shdrpgm();
}

/*  _________________________________________________________________________ */
/*!

@param none

@return none

*/
void GLApp::update() 
{
	glClearColor(std::abs( static_cast<float>( glm::sin(glfwGetTime() * 0.75f) ) ), std::abs( static_cast<float>( glm::cos(glfwGetTime() * 0.5f) ) ), 0.0f, 1.0f);
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
	mdl.draw();
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
/*! setup_vao

@param none

@return none

send vertex attribute data from cpu over to gpu and bind it to vaoid
*/
void GLApp::GLModel::setup_vao(void)
{
	// position vertex
	std::array<glm::vec2, 4> pos_vtx
	{
		glm::vec2(0.5f, -0.5f), glm::vec2(0.5f, 0.5f),
		glm::vec2(-0.5f, 0.5f), glm::vec2(-0.5f, -0.5f)
	};
	// color vertex
	std::array<glm::vec3, 4> clr_vtx
	{
		glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f),
		glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 1.f, 1.f)
	};

	GLuint vbo_hdl; glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	glCreateVertexArrays(1, &vaoid);
	// position vertex
	glEnableVertexArrayAttrib(vaoid, 8);
	glVertexArrayVertexBuffer(vaoid, 3, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vaoid, 8, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 8, 3);

	// color vertex
	glEnableVertexArrayAttrib(vaoid, 9);
	glVertexArrayVertexBuffer(vaoid, 4, vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(vaoid, 9, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 9, 4);

	primitive_type = GL_TRIANGLES;

	std::array<GLushort, 6> idx_vtx
	{
		0, 1, 2,
		2, 3, 0
	};

	idx_elem_cnt = idx_vtx.size();

	GLuint ebo_hdl; glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * idx_elem_cnt, reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vaoid, ebo_hdl);
	glBindVertexArray(0);
}

/*  _________________________________________________________________________ */
/*! setup_shdrpgm

@param none

@return none

initialize shdr_pgm to have a shader id for the program to draw
*/
void GLApp::GLModel::setup_shdrpgm()
{
	std::vector<std::pair<GLenum, std::string>> shdr_files;
	shdr_files.emplace_back( std::make_pair( GL_VERTEX_SHADER, "../shaders/my-tutorial-1.vert" ) );
	shdr_files.emplace_back( std::make_pair( GL_FRAGMENT_SHADER, "../shaders/my-tutorial-1.frag" ) );
	shdr_pgm.CompileLinkValidate(shdr_files);
	if (GL_FALSE == shdr_pgm.IsLinked())
	{
		std::cerr << "Unable to compile /link/validate shader programs" << std::endl;
		std::cerr << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

/*  _________________________________________________________________________ */
/*! draw

@param none

@return none

uses shader program and draw a rectangle based on the vaoid using triangle index
*/
void GLApp::GLModel::draw()
{
	shdr_pgm.Use();
	glBindVertexArray(vaoid);
 	glDrawElements(primitive_type, idx_elem_cnt, GL_UNSIGNED_SHORT, NULL);
	glBindVertexArray(0);
	shdr_pgm.UnUse();

	std::ostringstream oss;
	static double time = 1.0;
	time += GLHelper::delta_time;
	if (1.0 < time)
	{
		double fps = GLHelper::delta_time < 0.0001 ? 0.0 : 1.0 / GLHelper::delta_time;
		oss << std::fixed << std::setprecision(2) << "Tutorial 1 | Wong Man Cong | " << fps << std::endl;
		time = 0.0;
		glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());
	}
}