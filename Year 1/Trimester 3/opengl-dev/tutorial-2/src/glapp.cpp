/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		02/06/2022

This files implement functions that splits the viewport into 4 different regions
while rendering with different primitive types in each quadrant. The primitive
types used to render each quadrants are: GL_POINTS, GL_LINES, GL_TRIANGLE_FAN and GL_TRIANGLE_STRIP

Bonus task:
The bottom right quadrant that is rendered with GL_TRIANGLE_STRIP and glPrimitiveRestartIndex. 
This is implemented in three different functions.
1) glEnable(GL_PRIMITIVE_RESTART) is called inside the init function (Line 67)
2) An additional bookmark index is pushed into the idx_vtx after every triangle is formed
This is added inside the function tristrip_model (Line 379)
3) Before a call to glDrawElements, a call to glPrimitiveRestartIndex is called
inside the draw function (Line 455)
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#define _USE_MATH_DEFINES
#include <math.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
std::vector<GLApp::GLModel> GLApp::models;
std::vector<GLApp::GLViewport> GLApp::vps; 

double Random(double min, double max)
{
	double rand_max = static_cast<double>(RAND_MAX);
	double rand_val = static_cast<double>(rand());
	return (rand_val / rand_max) * (max - min) + min;
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

	srand(static_cast<unsigned int>(time(NULL)));

	GLint w{ GLHelper::width >> 1 }, h{ GLHelper::height >> 1 }; vps.reserve(4);
	vps.push_back( { 0, h, w, h } ); // Top left
	vps.push_back( { w, h, w, h } ); // Top right
	vps.push_back( { 0, 0, w, h } ); // Btm left
	vps.push_back( { w, 0, w, h } ); // Btm right

	models.reserve(4);
	GLApp::models.emplace_back(GLApp::points_model(20, 20, "../shaders/my-tutorial-2.vert", "../shaders/my-tutorial-2.frag"));	 // points
	GLApp::models.emplace_back(GLApp::lines_model(40, 40, "../shaders/my-tutorial-2.vert", "../shaders/my-tutorial-2.frag"));	 // lines
	GLApp::models.emplace_back(GLApp::trifans_model(50, "../shaders/my-tutorial-2.vert", "../shaders/my-tutorial-2.frag"));		 // circle
	GLApp::models.emplace_back(GLApp::tristrip_model(10, 15, "../shaders/my-tutorial-2.vert", "../shaders/my-tutorial-2.frag")); // tristrip

	glEnable(GL_PRIMITIVE_RESTART);
}

/*  _________________________________________________________________________ */
/*! update

@param none

@return none

empty for now (no animation for this tutorial)
*/
void GLApp::update() 
{
	
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

	for (size_t i = 0; i < 4; ++i)
	{
		glViewport(vps[i].x, vps[i].y, vps[i].width, vps[i].height);
		GLApp::models[i].draw();
	}

	std::ostringstream oss;
	static double time = 1.0;
	time += GLHelper::delta_time;
	if (1.0 < time)
	{
		double fps = GLHelper::delta_time < 0.0001 ? 0.0 : 1.0 / GLHelper::delta_time;
		oss << std::fixed << std::setprecision(2) << "Tutorial 2 | Wong Man Cong | "
			"POINTS: "		<< GLApp::models[0].primitive_cnt << ", " << GLApp::models[0].draw_cnt <<
			" | LINES: "	<< GLApp::models[1].primitive_cnt << ", " << GLApp::models[1].draw_cnt <<
			" | FAN: "		<< GLApp::models[2].primitive_cnt << ", " << GLApp::models[2].draw_cnt <<
			" | STRIP: "	<< GLApp::models[3].primitive_cnt << ", " << GLApp::models[3].draw_cnt << 
			" | " << fps;
		time = 0.0;
		glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());
	}
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
/*! cleanup

@param	slices : number of points to be generated horizontally
		stacks : number of points to be generated vertically
		vtx_shdr: file path to vertex shader
		frg_shdr: file path to fragment shader

Send vertex attribute data that forms points on the screen from cpu to the gpu and binds the buffer object to vaoid
*/
GLApp::GLModel GLApp::points_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr)
{
	size_t const max = static_cast<size_t>((slices + 1) * (stacks + 1));
	std::vector<glm::vec2> pos_vtx{ max };

	float const w = 2.0f / static_cast<float>(slices);
	float const h = 2.0f / static_cast<float>(stacks);
	for (size_t index = 0, i = 0; index < max; ++i)
	{
		for (size_t j = 0; j < static_cast<size_t>(stacks + 1); ++j)
			pos_vtx[index++] = glm::vec2{ (w * static_cast<float>(i)) - 1.0f, (h * static_cast<float>(j)) - 1.0f };
	}

	std::vector<glm::vec3> clr_vtx{ max };
	for (size_t i = 0; i < max; ++i)
	{
		float const r = static_cast<float>(Random(0.0, 1.0)), g = static_cast<float>(Random(0.0, 1.0)), b = static_cast<float>(Random(0.0, 1.0));
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vbo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	// position attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 1);

	glBindVertexArray(0);

	mdl.primitive_type = GL_POINTS;
	mdl.setup_shdrpgm(vtx_shdr, frg_shdr);
	mdl.primitive_cnt = mdl.draw_cnt = pos_vtx.size();
	return mdl;
}

/*  _________________________________________________________________________ */
/*! lines_model

@param	slices : number of lines to be generated horizontally
		stacks : number of lines to be generated vertically
		vtx_shdr: file path to vertex shader
		frg_shdr: file path to fragment shader

Send vertex attribute information to form a line from cpu to the gpu and binds the buffer object to vaoid
*/
GLApp::GLModel GLApp::lines_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr)
{
	size_t const max{ static_cast<size_t>( ( (slices + 1) * (stacks + 1) ) << 1) };
	std::vector<glm::vec2>pos_vtx{ max };
	float const w{ 2.0f / static_cast<float>(slices) };
	float const h{ 2.0f / static_cast<float>(stacks) };
	size_t index = 0;
	for (size_t i = 0; i < static_cast<size_t>(slices + 1); ++i)
	{
		float const x{ w * static_cast<float>(i) - 1.0f };
		pos_vtx[index++] = glm::vec2{ x, -1.0f };
		pos_vtx[index++] = glm::vec2{ x,  1.0f };
	}
	for (size_t i = 0; i < static_cast<size_t>(slices + 1); ++i)
	{
		float const y{ h * static_cast<float>(i) - 1.0f };
		pos_vtx[index++] = glm::vec2{ -1.0f, y };
		pos_vtx[index++] = glm::vec2{  1.0f, y };
	}

	std::vector<glm::vec3> clr_vtx{ max };
	for (size_t i = 0; i < max; ++i)
	{
		float const r = static_cast<float>(Random(0.0, 1.0)), g = static_cast<float>(Random(0.0, 1.0)), b = static_cast<float>(Random(0.0, 1.0));
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vbo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	// position attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 1);

	glBindVertexArray(0);

	mdl.primitive_type = GL_LINES;
	mdl.setup_shdrpgm(vtx_shdr, frg_shdr);
	mdl.draw_cnt = ((slices + 1) << 1) + ((stacks + 1) << 1); // number of vertices
	mdl.primitive_cnt = mdl.draw_cnt >> 1;
	return mdl;
}

/*  _________________________________________________________________________ */
/*! trifans_model

@param	slices : total number of slices to form a circle
		vtx_shdr: file path to vertex shader
		frg_shdr: file path to fragment shader

Send vertex attribute information to form a circle from cpu to the gpu and binds the buffer object to vaoid
*/
GLApp::GLModel GLApp::trifans_model(GLint slices, std::string const& vtx_shdr, std::string const& frg_shdr)
{
	size_t const max{ static_cast<size_t>(slices + 2) };
	std::vector<glm::vec2> pos_vtx{ max };
	size_t index = 0;
	pos_vtx[index++] = glm::vec2{ 0.0f, 0.0f };
	float const rad = (360.0f / static_cast<float>(slices)) * (static_cast<float>(M_PI) / 180.0f);
	for (size_t i = 0; i < static_cast<size_t>(slices + 1); ++i)
	{
		float const x = std::cosf(rad * i), y = std::sinf(rad * i);
		pos_vtx[index++] = glm::vec2{ x, y };
	}

	std::vector<glm::vec3> clr_vtx{ max };
	for (size_t i = 0; i < max; ++i)
	{
		float const r = static_cast<float>(Random(0.0, 1.0)), g = static_cast<float>(Random(0.0, 1.0)), b = static_cast<float>(Random(0.0, 1.0));
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vbo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	// position attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 1);

	glBindVertexArray(0);

	mdl.primitive_type = GL_TRIANGLE_FAN;
	mdl.setup_shdrpgm(vtx_shdr, frg_shdr);
	mdl.draw_cnt = pos_vtx.size();
	mdl.primitive_cnt = slices;
	return mdl;
}

/*  _________________________________________________________________________ */
/*! tristrip_model

@param	slices : number of points to be generated horizontally
		stacks : number of points to be generated vertically
		vtx_shdr: file path to vertex shader
		frg_shdr: file path to fragment shader

Send vertex attribute information to form a line from cpu to the gpu and binds the buffer object to vaoid
*/
GLApp::GLModel GLApp::tristrip_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr)
{
	size_t const max = static_cast<size_t>((slices + 1) * (stacks + 1));
	std::vector<glm::vec2> pos_vtx{ max };

	float const w = 2.0f / static_cast<float>(slices);
	float const h = 2.0f / static_cast<float>(stacks);
	for (size_t index = 0, i = 0; index < max; ++i)
	{
		for (size_t j = 0; j < static_cast<size_t>(slices + 1); ++j)
			pos_vtx[index++] = glm::vec2{ (w * static_cast<float>(j)) - 1.0f, (h * static_cast<float>(i)) - 1.0f };
	}

	std::vector<glm::vec3> clr_vtx{ max };
	for (size_t i = 0; i < max; ++i)
	{
		float const r = static_cast<float>(Random(0.0, 1.0)), g = static_cast<float>(Random(0.0, 1.0)), b = static_cast<float>(Random(0.0, 1.0));
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vbo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	// position attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 1);

	glBindVertexArray(0);

	size_t const slices_by_1{ static_cast<size_t>(slices + 1) }, slices_by_2{ static_cast<size_t>(slices + 2) };
	size_t const max_idx{ static_cast<size_t>(slices + 1) << 1 };
	std::vector<GLushort> idx_vtx; 
	for (size_t h{ 0 }; h < static_cast<size_t>(stacks); ++h)
	{
		idx_vtx.push_back( static_cast<GLushort>(slices_by_1 * (h + 1)) );
		for (size_t w{ 0 }; w < static_cast<size_t>(slices + 1); ++w)
		{
			idx_vtx.push_back( static_cast<GLushort>(idx_vtx.back() - slices_by_1) );
			idx_vtx.push_back( static_cast<GLushort>(idx_vtx.back() + slices_by_2) );
		}
		idx_vtx.pop_back(); idx_vtx.push_back(GL_PRIMITIVE_RESTART_INDEX);
	}

	GLuint ebo; glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLuint) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(mdl.vaoid, ebo);

	mdl.primitive_type = GL_TRIANGLE_STRIP;
	mdl.setup_shdrpgm(vtx_shdr, frg_shdr);
	mdl.draw_cnt = idx_vtx.size();
	mdl.primitive_cnt = (slices * stacks) << 1;
	return mdl;
}

/*  _________________________________________________________________________ */
/*! setup_shdrpgm

@param 	vtx_shdr: file path to vertex shader
		frg_shdr: file path to fragment shader

@return none

initialize shdr_pgm to have a shader id for the program to draw
*/
void GLApp::GLModel::setup_shdrpgm(std::string const& vtx_shdr, std::string const& frg_shdr)
{
	std::vector<std::pair<GLenum, std::string>> shdr_files;
	shdr_files.emplace_back( std::make_pair( GL_VERTEX_SHADER, vtx_shdr ) );
	shdr_files.emplace_back( std::make_pair( GL_FRAGMENT_SHADER, frg_shdr ) );
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

uses shader program to render different shapes based on their primitive_type
*/
void GLApp::GLModel::draw()
{
	shdr_pgm.Use();
	glBindVertexArray(vaoid);
	switch (primitive_type)
	{
		case GL_POINTS:
		{
			glPointSize(10.0f);
			//glVertexAttrib3f(1, 1.0f, 0.0f, 0.0f);
			glDrawArrays(primitive_type, 0, draw_cnt);
			glPointSize(1.0f);
			break;
		}
		case GL_LINES:
		{
			glLineWidth(3.f);
			//glVertexAttrib3f(1, 0.f, 0.f, 1.f); 
			glDrawArrays(primitive_type, 0, draw_cnt);
			glLineWidth(1.f);
			break;
		}
		case GL_TRIANGLE_FAN:
		{
			glDrawArrays(primitive_type, 0, draw_cnt);
			break;
		}
		case GL_TRIANGLE_STRIP:
		{
			GLushort const index = static_cast<GLushort>(GL_PRIMITIVE_RESTART_INDEX);
			glPrimitiveRestartIndex(index);
			glDrawElements(primitive_type, draw_cnt, GL_UNSIGNED_SHORT, nullptr);
			break;
		}
		default:
			break;
	}
	glBindVertexArray(0);
	shdr_pgm.UnUse();
}