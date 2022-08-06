/* !
@file       glpbo.cpp
@author     pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date       08/07/2022

This file contains function implementations that renders line using the bresenham
algorithm and using the triangle edge equation to determine if a pixel is inside
the plane or not, thus rendering a triangle as a result. This is all done on the
client side (CPU).

The following keys are used to interact with the program:
W: To switch between the different modes of rendering
R: To toggle between if the object should be rotating
M: To change between the different models loaded into the project
G: To toggle between the models loaded using DigiPen's Model Loader and my own
   custom .obj paser
*//*__________________________________________________________________________*/
#include "glpbo.h"
double fps = 0.0;

// static variables
GLsizei GLPbo::width = 0, GLPbo::height = 0;
GLsizei GLPbo::pixel_cnt = 0, GLPbo::byte_cnt = 0;
GLPbo::Color *GLPbo::ptr_to_pbo = nullptr;
GLuint GLPbo::vaoid, GLPbo::elem_cnt, GLPbo::pboid, GLPbo::texid;
GLSLShader GLPbo::shdr_pgm;
GLPbo::Color GLPbo::clear_clr;
GLPbo::Object GLPbo::objs[static_cast<size_t>(GLPbo::ObjectTypes::Total)][2];

// global variables
namespace
{
	glm::mat4 vp_mtx;
	size_t obj_index;
	bool r_key_pressed, w_key_pressed, m_key_pressed, g_key_pressed;

	float constexpr rotating_speed = 45.0f;
	size_t constexpr max_rm = static_cast<size_t>(GLPbo::RenderMode::Total),
					 max_obj = static_cast<size_t>(GLPbo::ObjectTypes::Total);

	// use to differenitiate between the different models loaded by my own paser and DPML's parser
	size_t parser_index;

	/*  _________________________________________________________________________ */
	/*! Random

	@param	min: lower bound
			max: upper bound

	@return a random number

	Randomizes a number between min and max
	*/
	int Random(int min, int max)
	{
		std::random_device rd; std::mt19937 gen(rd()); std::uniform_int_distribution<int> dis(min, max);
		return dis(gen);
	}
}

/*  _________________________________________________________________________ */
/*!	emulate

@param	none

@return none

get a pointer to the pixel buffer object and changes the color every frame
*/
void GLPbo::emulate()
{
	ptr_to_pbo = static_cast<Color*>(glMapNamedBuffer(pboid, GL_WRITE_ONLY));
	// clear color buffer
	clear_color_buffer();

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_R) && !r_key_pressed)
	{
		objs[obj_index][parser_index].rotating = !objs[obj_index][parser_index].rotating;
		r_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_R) && r_key_pressed)
		r_key_pressed = false;

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_W) && !w_key_pressed)
	{
		size_t index = static_cast<size_t>(objs[obj_index][parser_index].rm); (++index) %= max_rm;
		objs[obj_index][parser_index].rm = static_cast<RenderMode>(index);
		w_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_W) && w_key_pressed)
		w_key_pressed = false;

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_M) && !m_key_pressed)
	{
		(++obj_index) %= max_obj;
		m_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_M) && m_key_pressed)
		m_key_pressed = false;

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_G) && !g_key_pressed)
	{
		(++parser_index) %= 2;
		g_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_G) && g_key_pressed)
		g_key_pressed = false;

	if (objs[obj_index][parser_index].rotating)
		objs[obj_index][parser_index].angle += rotating_speed * static_cast<float>(GLHelper::delta_time);

	viewport_xform();

	switch (objs[obj_index][parser_index].rm)
	{
		case RenderMode::Wireframe: case RenderMode::Wireframe_colored:
		{
			wireframe_mode();
			break;
		}
		case RenderMode::Faceted: case RenderMode::Shaded:
		{
			triangle_mode();
			break;
		}
		default:
			break;
	}

	glUnmapNamedBuffer(pboid);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboid);
	glTextureSubImage2D(texid, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

/*  _________________________________________________________________________ */
/*! draw_fullwindow_quad

@param	none

@return none

draw function to see color changes on the screen every frame
*/
void GLPbo::draw_fullwindow_quad()
{
	std::ostringstream oss;
	static double time = 1.0;
	time += GLHelper::delta_time;

	if (1.0 < time)
	{
		fps = GLHelper::delta_time < 0.0001 ? 0.0 : 1.0 / GLHelper::delta_time;
		time = 0.0;
	}

	std::string model = "Model: ", mode = "Mode: ", paser = "Paser: ";
	Object const& obj = objs[obj_index][parser_index];

	switch (obj.rm)
	{
		case RenderMode::Wireframe:
		{
			mode += "Wireframe";
			break;
		}
		case RenderMode::Wireframe_colored:
		{
			mode += "Wireframe Color";
			break;
		}
		case RenderMode::Faceted:
		{
			mode += "Faceted";
			break;
		}
		case RenderMode::Shaded:
		{
			mode += "Shaded";
			break;
		}
	}

	switch (static_cast<ObjectTypes>(obj_index))
	{
		case ObjectTypes::Cube:
		{
			model += "Cube";
			break;
		}
		case ObjectTypes::Suzanne:
		{
			model += "Suzanne";
			break;
		}
		case ObjectTypes::Ogre:
		{
			model += "Ogre";
			break;
		}
		case ObjectTypes::Head:
		{
			model += "Head";
			break;
		}
		case ObjectTypes::Teapot:
		{
			model += "Teapot";
			break;
		}
	}

	paser += parser_index ? "Custom" : "DPML";

	oss << std::fixed << std::setprecision(2) << "A1 | Wong Man Cong | " << model << " | " << mode << " | " << paser <<
		" | Vertices: " << obj.pos.size() <<
		" | Triangles: " << obj.pos_idx.size() / 3 <<
		" | Culled: " << obj.culled << 
		" | FPS: " << fps;

	glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());

	glBindTextureUnit(0, texid);
	glTextureParameteri(texid, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTextureParameteri(texid, GL_TEXTURE_WRAP_T, GL_REPEAT);

	shdr_pgm.Use();

	GLuint tex_loc = glGetUniformLocation(shdr_pgm.GetHandle(), "uTex2d");
	glUniform1i(tex_loc, 0);

	glBindVertexArray(vaoid);
	glDrawElements(GL_TRIANGLES, elem_cnt, GL_UNSIGNED_SHORT, nullptr);
	glBindVertexArray(0);

	shdr_pgm.UnUse();
}

/*  _________________________________________________________________________ */
/*! init

@param	w: width of glfw window
		h: height of glfw window

@return none

Initialisation of GLPbo application
*/
void GLPbo::init(GLsizei w, GLsizei h)
{
	pixel_cnt = w * h, byte_cnt = pixel_cnt * 4;
	width	  = w, height = h;

	set_clear_color(255, 255, 255, 255);

	glCreateTextures(GL_TEXTURE_2D, 1, &texid);
	glTextureStorage2D(texid, 1, GL_RGBA8, width, height);

	glCreateBuffers(1, &pboid);
	glNamedBufferStorage(pboid, byte_cnt, nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);

	setup_quad_vao();
	setup_shdrpgm();

	obj_index = parser_index = 0;

	r_key_pressed = w_key_pressed = m_key_pressed = g_key_pressed = false;

	glEnable(GL_SCISSOR_TEST);
	glScissor(0, 0, width, height);

	viewport_mtx();

	load_scene();
}

/*  _________________________________________________________________________ */
/*! setup_quad_vao

@param	none

@return none

create a buffer object to render a quad and bind it to vaoid 
*/
void GLPbo::setup_quad_vao()
{
	std::vector<glm::vec2> pos_vtx
	{
		glm::vec2(-1.0f, -1.0f), glm::vec2(1.0f, -1.0f),
		glm::vec2( 1.0f,  1.0f), glm::vec2(-1.0f, 1.0f)
	}, tex_vtx
	{
		glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f), glm::vec2(0.0f, 1.0f)
	};

	GLuint vbo = 0, ebo = 0;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec2) * tex_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec2) * tex_vtx.size(), tex_vtx.data());

	glCreateVertexArrays(1, &vaoid);
	// position attribute
	glEnableVertexArrayAttrib(vaoid, 0);
	glVertexArrayVertexBuffer(vaoid, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 0, 0);

	// texture attribute
	glEnableVertexArrayAttrib(vaoid, 1);
	glVertexArrayVertexBuffer(vaoid, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec2));
	glVertexArrayAttribFormat(vaoid, 1, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 1, 1);

	// ebo
	std::vector<GLushort> idx_vtx{ 1, 2, 0, 2, 3, 0 };
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLushort) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vaoid, ebo);

	elem_cnt = idx_vtx.size();

	// unbind vao to prevent unwanted changes
	glBindVertexArray(0);
}

/*  _________________________________________________________________________ */
/*! setup_shdrpgm

@param	none

@return none

using c++ string to setup a simple shader program for vertex and fragment
*/
void GLPbo::setup_shdrpgm()
{
	std::string vtx =
		"#version 450 core\n"
		"layout(location = 0) in vec2 aPos;\n"
		"layout(location = 1) in vec2 aTexCoords;\n"
		"layout(location = 0) out vec2 vTexCoords;\n"
		"void main()\n"
		"{\n"
		"gl_Position = vec4(aPos, 0.0f, 1.0f);\n"
		"vTexCoords = aTexCoords;\n"
		"}\n";
	std::string frg =
		"#version 450 core\n"
		"layout(location = 0) in vec2 vTexCoords;\n"
		"layout(location = 0) out vec4 fColor;\n"
		"uniform sampler2D uTex2d;\n"
		"void main()\n"
		"{\n"
		"fColor = texture(uTex2d, vTexCoords);\n"
		"}\n";

	shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER,	 vtx);
	shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, frg);
	shdr_pgm.Link();
	shdr_pgm.Validate();
}

/*  _________________________________________________________________________ */
/*! cleanup

@param	none

@return none

return resources back to gpu
*/
void GLPbo::cleanup()
{
	glDeleteVertexArrays(1, &vaoid);
	glDeleteBuffers(1, &pboid);
	glDeleteTextures(1, &texid);
}

/*  _________________________________________________________________________ */
/*!	set_clear_color

@param	clr: value containing the new color

@return none

set the value for clear color
*/
void GLPbo::set_clear_color(GLPbo::Color clr)
{
	clear_clr = clr;
}

/*  _________________________________________________________________________ */
/*! set_clear_color

@param	r: unsigned value for red color
		g: unsigned value for green	color
		b: unsigned value for blue color
		a: unsigned value for alpha

@return none

set the value for clear_color
*/
void GLPbo::set_clear_color(GLubyte r, GLubyte g, GLubyte b, GLubyte a)
{
	clear_clr.r = r, clear_clr.g = g, clear_clr.b = b, clear_clr.a = a;
}

/*  _________________________________________________________________________ */
/*! clear_color_buffer

@param	none

@return none

function similar to glClear which fills the buffer with the specified clear_clr
*/
void GLPbo::clear_color_buffer()
{
	std::fill_n(ptr_to_pbo, pixel_cnt, clear_clr);
}

/*  _________________________________________________________________________ */
/*! load_scene

@param	none

@return none

load the appropriate objects name in the scene into the project
*/
void GLPbo::load_scene()
{
	std::ifstream ifs{ "../scenes/ass-1.scn" };
	std::string buf, obj_name;
	size_t index = 0;

	while (std::getline(ifs, buf))
	{
		std::istringstream iss{ buf };
		iss >> obj_name;

		// loading using DPML
		if (!DPML::parse_obj_mesh("../meshes/" + obj_name + ".obj", objs[index][0].pos, objs[index][0].nml, objs[index][0].tex, objs[index][0].pos_idx, true, false))
		{
			std::cerr << "Unable to load: " << obj_name << ". File is either not present, unreadable or doesn't follow the OBJ file format" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		size_t const nml_size0 = objs[index][0].nml.size(), clr_size0 = objs[index][0].pos_idx.size() / 3;
		objs[index][0].pd  = std::vector<glm::vec3>(nml_size0);
		objs[index][0].clr = std::vector<Color>(clr_size0);

		for (size_t i = 0; i < nml_size0; ++i)
		{
			glm::vec3& normal = objs[index][0].nml[i];
			normal.x = (normal.x + 1.0f) * 0.5f;
			normal.y = (normal.y + 1.0f) * 0.5f;
			normal.z = (normal.z + 1.0f) * 0.5f;
		}

		for (size_t i = 0; i < clr_size0; ++i)
		{
			GLubyte r = static_cast<GLubyte>(Random(0, 255)), g = static_cast<GLubyte>(Random(0, 255)), b = static_cast<GLubyte>(Random(0, 255));
			objs[index][0].clr[i] = Color(r, g, b);
		}

		// loading using my own parser
		if (!my_own_obj_parser("../meshes/" + obj_name + ".obj", objs[index][1].pos, objs[index][1].nml, objs[index][1].tex, objs[index][1].pos_idx, true, false))
		{
			std::cerr << "Unable to load: " << obj_name << ". File is either not present, unreadable or doesn't follow the OBJ file format" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		size_t const nml_size1 = objs[index][1].nml.size(), clr_size1 = objs[index][1].pos_idx.size() / 3;
		objs[index][1].pd = std::vector<glm::vec3>(nml_size1);
		objs[index][1].clr = std::vector<Color>(clr_size1);

		for (size_t i = 0; i < nml_size1; ++i)
		{
			glm::vec3& normal = objs[index][1].nml[i];
			normal.x = (normal.x + 1.0f) * 0.5f;
			normal.y = (normal.y + 1.0f) * 0.5f;
			normal.z = (normal.z + 1.0f) * 0.5f;
		}

		for (size_t i = 0; i < clr_size1; ++i)
		{
			GLubyte r = static_cast<GLubyte>(Random(0, 255)), g = static_cast<GLubyte>(Random(0, 255)), b = static_cast<GLubyte>(Random(0, 255));
			objs[index][1].clr[i] = Color(r, g, b);
		}

		++index;
	}
}

/*  _________________________________________________________________________ */
/*! set_pixel_color

@param	x: x coordinate
		y: y coordinate
		clr: clr at (x,y) coordinate

@return none

set a particular color at (x,y) coordinate
*/
void GLPbo::set_pixel_color(size_t x, size_t y, Color clr)
{
	if (static_cast<size_t>(width) <= x || static_cast<size_t>(height) <= y)
		return;
	*(ptr_to_pbo + y * width + x) = clr;
}

/*  _________________________________________________________________________ */
/*! viewport_mtx

@param	none

@return none

helper function to generate a viewport mtx based on the window's width and height
*/
void GLPbo::viewport_mtx()
{
	vp_mtx = glm::mat4{ 1.0f };
	float const half_width = static_cast<float>(GLPbo::width >> 1), half_height = static_cast<float>(GLPbo::height >> 1);
	vp_mtx[0][0] = vp_mtx[2][0] = half_width;
	vp_mtx[1][1] = vp_mtx[2][1] = half_height;
}

/*  _________________________________________________________________________ */
/*! viewport_xform

@param	none

@return none

transform the ndc coordinates to window coordinate
*/
void GLPbo::viewport_xform()
{
	Object& obj = objs[obj_index][parser_index];
	size_t pos_size = obj.pos.size();
	// creating alias for obj's pos and pd
	std::vector<glm::vec3> const& pos = obj.pos; std::vector<glm::vec3>& pd = obj.pd;
	glm::mat4 rot{ 1.0f };
	rot = glm::rotate(rot, glm::radians(obj.angle), glm::vec3(0.0f, 0.0f, 1.0f));
	glm::mat4 mtx = vp_mtx * rot;

	// looping thru each obj's position and determine their device coordinate
	for (size_t i = 0; i < pos_size; ++i)
	{
		glm::vec4 device_pos = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
		device_pos.x = pos[i].x, device_pos.y = pos[i].y;
		pd[i] = mtx * device_pos, pd[i].z = 0.0f;
	}
}

/*  _________________________________________________________________________ */
/*! wireframe_mode

@param	none

@return none

using the bresenham algorithm, render lines on the screen
*/
void GLPbo::wireframe_mode()
{
	// total number of iterations
	size_t iterations = objs[obj_index][parser_index].pos_idx.size() / 3;

	std::vector<unsigned short> const& idx = objs[obj_index][parser_index].pos_idx;
	Object& obj = objs[obj_index][parser_index];
	obj.culled = 0;

	for (size_t i = 0; i < iterations; ++i)
	{
		glm::vec2 pos[3]{};
		pos[0] = obj.pd[ idx[i * 3] ];
		pos[1] = obj.pd[ idx[1 + i * 3] ];
		pos[2] = obj.pd[ idx[2 + i * 3] ];

		if (backface_cull(pos))
		{
			++obj.culled;
			continue;
		}

		Color const& clr = obj.rm == RenderMode::Wireframe ? Color(0, 0, 0, 255) : obj.clr[i];

		int const x0 = static_cast<int>(pos[0].x), y0 = static_cast<int>(pos[0].y);
		int const x1 = static_cast<int>(pos[1].x), y1 = static_cast<int>(pos[1].y);
		int const x2 = static_cast<int>(pos[2].x), y2 = static_cast<int>(pos[2].y);

		render_linebresenham(x0, y0, x1, y1, clr);
		render_linebresenham(x1, y1, x2, y2, clr);
		render_linebresenham(x2, y2, x0, y0, clr);
	}
}

/*  _________________________________________________________________________ */
/*! triangle_area

@param	pos: an array of 3 vec2 storing the window coordinate of the triangle

@return the area of the triangle

calculate the area of the triangle based on the position of the 3 vertices
*/
float GLPbo::triangle_area(glm::vec2 const pos[3])
{
	return (pos[1].x - pos[0].x) * (pos[2].y - pos[0].y) - (pos[2].x - pos[0].x) * (pos[1].y - pos[0].y);
}

/*  _________________________________________________________________________ */
/*!	triangle_area

@param	x0: x coordinate of vertex 0
		y0:	y coordinate of vertex 0
		x1:	x coordinate of vertex 1
		y1:	y coordinate of vertex 1
		x2:	x coordinate of vertex 2
		y2: y coordinate of vertex 2

@return the area of the triangle

calculate the area of the triangle based on the position of the vertices
*/
float GLPbo::triangle_area(float x0, float y0, float x1, float y1, float x2, float y2)
{
	glm::vec2 pos[3]{ glm::vec2(x0, y0), glm::vec2(x1, y1), glm::vec2(x2, y2) };
	return triangle_area(pos);
}

/*  _________________________________________________________________________ */
/*! backface_cull

@param	area: area of the triangle

@return true if area is positive, false if negative

checks if the area is positive or negative
*/
bool GLPbo::backface_cull(float area)
{
	return area < 0.0f;
}

/*  _________________________________________________________________________ */
/*! backface_cull

@param	pos: an array of 3 vec2 storing the window coordinate of the triangle

@return true if area is positive, false if negative

checks if the area is positive or negative
*/
bool GLPbo::backface_cull(glm::vec2 const pos[3])
{
	return backface_cull(triangle_area(pos));
}

/*  _________________________________________________________________________ */
/*! render_linebresenham

@param	x1:	x coordinate of vertex 1 in window coordinate
		y1:	y coordinate of vertex 1 in window coordinate
		x2:	x coordinate of vertex 2 in window coordinate
		y2: y coordinate of vertex 2 in window coordinate
		clr: clr to be set at the (x,y) coordinate

@return none

using the bresenham line algorithm, calculate the decision parameter and
render a line on the screen
*/
void GLPbo::render_linebresenham(int x1, int y1, int x2, int y2, Color clr)
{
	int dx = x2 - x1, dy = y2 - y1;

	if (!dx && !dy)
		return;

	int xstep = dx < 0 ? -1 : 1, ystep = dy < 0 ? -1 : 1;

	dx = dx < 0 ? -dx : dx;
	dy = dy < 0 ? -dy : dy;

	set_pixel_color(x1, y1, clr);

	if (dx >= dy)
	{
		int dk = (dy << 1) - dx, dmin = dy << 1, dmaj = (dy << 1) - (dx << 1);
		while (--dx)
		{
			y1 += dk > 0 ? ystep : 0;
			dk += dk > 0 ? dmaj : dmin;
			x1 += xstep;
			set_pixel_color(x1, y1, clr);
		}
	}
	else
	{
		int dk = (dx << 1) - dy, dmin = dx << 1, dmaj = (dx << 1) - (dy << 1);
		while (--dy)
		{
			x1 += dk > 0 ? xstep : 0;
			dk += dk > 0 ? dmaj : dmin;
			y1 += ystep;
			set_pixel_color(x1, y1, clr);
		}
	}
}

/*  _________________________________________________________________________ */
/*! triangle_mode

@param	none

@return none

using the triangle edge equation formula, determine if a particular 
point on screen should be rendered on screen and rendering a triangle as a result
*/
void GLPbo::triangle_mode()
{
	// total number of iterations
	size_t iterations = objs[obj_index][parser_index].pos_idx.size() / 3;

	std::vector<unsigned short> const& idx = objs[obj_index][parser_index].pos_idx;
	Object& obj = objs[obj_index][parser_index];
	obj.culled = 0;

	for (size_t i = 0; i < iterations; ++i)
	{
		glm::vec2 pos[3]{};
		pos[0] = obj.pd[idx[i * 3]];
		pos[1] = obj.pd[idx[1 + i * 3]];
		pos[2] = obj.pd[idx[2 + i * 3]];

		// area of the triangle
		float A = triangle_area(pos) * 0.5f;

		if (backface_cull(A))
		{
			++obj.culled;
			continue;
		}

		// Compute the edge equation of the triangle
		auto edge_equation = [](float x1, float y1, float x2, float y2)
		{
			Edge e;
			e.a = y1 - y2;
			e.b = x2 - x1;
			e.c = x1 * y2 - x2 * y1;
			e.tl = (e.a != 0.0f) ? (e.a > 0.0f) : (e.b < 0.0f);
			return e;
		};

		// Compute triangle equation
		Triangle tri;
		tri.e0 = edge_equation(pos[1].x, pos[1].y, pos[2].x, pos[2].y);
		tri.e1 = edge_equation(pos[2].x, pos[2].y, pos[0].x, pos[0].y);
		tri.e2 = edge_equation(pos[0].x, pos[0].y, pos[1].x, pos[1].y);

		// compute aabb
		glm::vec<2, size_t> min{ 0,0 }, max{ 0,0 };
		min.x = static_cast<size_t>( std::floor( std::min( pos[0].x, std::min(pos[1].x, pos[2].x) ) ) );
		min.y = static_cast<size_t>( std::floor( std::min( pos[0].y, std::min(pos[1].y, pos[2].y) ) ) );
		max.x = static_cast<size_t>( std::ceil ( std::max( pos[0].x, std::max(pos[1].x, pos[2].x) ) ) );
		max.y = static_cast<size_t>( std::ceil ( std::max( pos[0].y, std::max(pos[1].y, pos[2].y) ) ) );

		// evaulation of pixel to check if it's part of the triangle
		auto evaluation_value = [](Edge const& e, glm::vec2 const& p)
		{
			return e.a * p.x + e.b * p.y + e.c;
		};

		glm::vec2 pos0 = glm::vec2(static_cast<float>(min.x) + 0.5f, static_cast<float>(min.y) + 0.5f);
		float vEval0 = evaluation_value( tri.e0, pos0 );
		float vEval1 = evaluation_value( tri.e1, pos0 );
		float vEval2 = evaluation_value( tri.e2, pos0 );

		// Check if the point is in the edge
		auto point_in_edge = [](float eval, bool tl)
		{
			return eval > 0.0f || (eval == 0 && tl);
		};

		// Check if the point is in the triangle
		auto point_in_triangle = [&](float hEval0, float hEval1, float hEval2)
		{
			return point_in_edge(hEval0, tri.e0.tl) && point_in_edge(hEval1, tri.e1.tl) && point_in_edge(hEval2, tri.e2.tl);
		};

		// Color will be used if RenderMode is not shaded
		Color const& clr = obj.clr[i];

		// Calculate the new area to use for the calculation of color when linearly interpolating thru the vertices
		A = 1.0f / (2.0f * A);

		for (size_t y = min.y; y < max.y; ++y)
		{
			float hEval0 = vEval0, hEval1 = vEval1, hEval2 = vEval2;
			for (size_t x = min.x; x < max.x; ++x)
			{
				if (point_in_triangle(hEval0, hEval1, hEval2))
				{
					switch (obj.rm)
					{
						case RenderMode::Faceted:
						{
							set_pixel_color(x, y, clr);
							break;
						}
						case RenderMode::Shaded:
						{
							// Calculate the area of the triangle
							float const xd = static_cast<float>(x), yd = static_cast<float>(y);
							float const A0 = triangle_area(xd, yd, pos[1].x, pos[1].y, pos[2].x, pos[2].y) * A;
							float const A1 = triangle_area(xd, yd, pos[2].x, pos[2].y, pos[0].x, pos[0].y) * A;
							float const A2 = triangle_area(xd, yd, pos[0].x, pos[0].y, pos[1].x, pos[1].y) * A;

							glm::vec3 nml[3]{};
							nml[0] = obj.nml[idx[i * 3]] * A0;
							nml[1] = obj.nml[idx[1 + i * 3]] * A1;
							nml[2] = obj.nml[idx[2 + i * 3]] * A2;

							nml[0] += nml[1] + nml[2];
							GLubyte r = static_cast<GLubyte>(nml[0].x * 255.0f),
								g = static_cast<GLubyte>(nml[0].y * 255.0f),
								b = static_cast<GLubyte>(nml[0].z * 255.0f);

							set_pixel_color(x, y, Color(r, g, b, 255));

							break;
						}
					}
				}
				hEval0 += tri.e0.a, hEval1 += tri.e1.a, hEval2 += tri.e2.a;
			}
			vEval0 += tri.e0.b, vEval1 += tri.e1.b, vEval2 += tri.e2.b;
		}
	}
}

/*  _________________________________________________________________________ */
/*! my_own_obj_parser

  @param std::string filename
  The name of the file containing the OBJ geometry information.

  @param std::vector<glm::vec3>& positions
  Fill user-supplied container with vertex position attributes.

  @param std::vector<glm::vec3>& normals
  Fill user-supplied container with vertex normal attributes.
  The container will not be touched if parameter "nml_attribs_flag" is
  false.

  @param std::vector<glm::vec2>& texcoords
  Fill user-supplied container with vertex texture coordinate attributes.
  The container will not be touched if parameter "texcoords_attribs_flag" is
  false.

  @param std::vector<unsigned short>& triangles
  Triangle vertices are specified as indices into containers "positions",
  "normals", and "texcoords". Triangles will always have counter-clockwise
  orientation. This means that when looking at a face from the outside of
  the box, the triangles are counter-clockwise oriented.
  Use an indexed draw call to draw the box.

  @param bool load_tex_coord_flag = false
  If parameter is true, then texture coordinates (if present in file) will
  be parsed. Otherwise, texture coordinate (even if present in file) will
  not be read.

  @param bool load_nml_coord_flag = false
  If parameter is true, then per-vertex normal coordinates (if present in file)
  will be parsed if they are present in file, otherwise, the per-vertex
  normals are computed.
  If the parameter is false, normal coordinate will neither be read from
  file (if present) nor explicitly computed.

  @param bool model_centered_flag = true
  In some cases, the modeler might have generated the model such that the
  center (of gravity) of the model is not centered at the origin.
  If the parameter is true, then the function will compute an axis-aligned
  bounding box and translate the position coordinates so that the box's center
  is at the origin.
  If the parameter is false, the position coordinates are left untouched.

  @return bool
  true if successful, otherwise false.
  The function can return false if the file is not present

  This function parses an OBJ geometry file and stores the contents of the file
  as array of vertex, array of normal (if required), and an array of texture
  (if required) coordinate data. These three arrays will have the same size.
  Triangles are defined as an array of indices into array of position
  coordinates.
*/
bool GLPbo::my_own_obj_parser(std::string filename,
	std::vector<glm::vec3>& positions,
	std::vector<glm::vec3>& normals,
	std::vector<glm::vec2>& texcoords,
	std::vector<unsigned short>&  position_indices,
	bool                          load_nml_coord_flag,
	bool                          load_tex_coord_flag,
	bool                          model_centered_flag)
{
	std::ifstream ifs{ filename };
	std::string buf, line;

	if (ifs.fail())
	{
		std::cerr << "Unable to parse " << filename << std::endl;
		return false;
	}

	glm::vec2 min(0.0f), max(0.0f);
	// clear the vectors
	positions.clear(), normals.clear(), texcoords.clear(), position_indices.clear();

	while (std::getline(ifs, buf))
	{
		std::istringstream iss{ buf };

		iss >> line;
		// position attribute
		if (line == "v")
		{
			glm::vec3 pos(0.0f);
			iss >> pos.x >> pos.y >> pos.z;

			if (pos.x < min.x)
				min.x = pos.x;
			if (pos.y < min.y)
				min.y = pos.y;

			if (pos.x > max.x)
				max.x = pos.x;
			if (pos.y > max.y)
				max.y = pos.y;

			positions.push_back(pos);
		}
		// texture coord attribute
		else if (line == "vt" && load_tex_coord_flag)
		{
			glm::vec2 tex(0.0f);
			iss >> tex.x >> tex.y;
			texcoords.push_back(tex);
		}
		// normal attribute
		else if (line == "vn" && load_nml_coord_flag)
		{
			glm::vec3 nml(0.0f);
			iss >> nml.x >> nml.y >> nml.z;
			normals.push_back(nml);
		}
		// indices
		else if (line == "f")
		{
			// contain the faces of the model
			// in normal cases, i would need to have 3 different vectors to store the indices of position, texcoord and normals
			// but for this assignment, just need to store indices of position
			for (size_t i = 0; i < 3; ++i)
			{
				iss >> line;
				size_t last_index = line.find_first_of('/');
				position_indices.push_back( static_cast<unsigned short>( std::stoi( line.substr(0, last_index) ) - 1) );
			}
		}
	}

	// Translate the model to the center
	if (model_centered_flag)
	{
		glm::vec3 const offset = ( glm::vec3{ max.x, max.y, 0.0f } + glm::vec3{ min.x, min.y,0.0f } ) * 0.5f; size_t const pos_size = positions.size();
		for (size_t i = 0; i < pos_size; ++i)
			positions[i] -= offset;
	}

	// Calculate normal myself if normal vector is empty
	if (!normals.size())
	{
		normals = std::vector<glm::vec3>(positions.size());
		size_t const nml_size = normals.size(), idx_size = position_indices.size();
		for (size_t i = 0; i < nml_size; ++i)
			normals[i] = glm::vec3(0.0f);

		using u16 = unsigned short;
		for (size_t i = 0; i < idx_size; i += 3)
		{
			u16 const i0 = position_indices[i], 
				i1 = position_indices[i + 1],
				i2 = position_indices[i + 2];

			glm::vec3 const e0 = positions[i1] - positions[i0];
			glm::vec3 const e1 = positions[i2] - positions[i0];
			glm::vec3 const no = glm::cross(e0, e1);

			normals[i0] += no; normals[i1] += no; normals[i2] += no;
		}

		for (size_t i = 0; i < nml_size; ++i)
			normals[i] = glm::normalize(normals[i]);
	}

	return true;
}