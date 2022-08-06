/* !
@file       glpbo.cpp
@author     pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date       16/07/2022

This file contains function definition for rendering a 3D model of an ogre and
a cube. It uses the bresenham algorithm to render the models in wireframe mode
and triangle edge equation to determine if a fragment should exist in a triangle
when rendering for triangles. This program have 7 different rendering mode.

Wireframe: 
- Render model using lines

Depth Buffer:
- Interpolates the depth buffer and use it as a color value

Faceted: 
- Calculates illumination based on face's normal

Shaded:
- Calculate illumination of each vertex normal and interpolate the color

Texture:
- Map each texel onto the model

Texture/Faceted: 
- Same as Faceted, but will use the texel at (xd, yd) as a diffuse color

Texture/Shaded:
- Same as Shaded, but will use the texel at (xd, yd) as a diffuse color

The keys that are used for interaction in the program are:
W: To switch between the different render mode
R: To toggle rotation of a model
M: Iterates through the different models loaded into the program
X: Rotate about axis (1.0f, 1.0f, 0.0f)
Z: Rotate about axis (0.0f, 1.0f, 1.0f)

When both x and z is toggled, the model will rotate about axis (1.0f, 1.0f, 1.0f)

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
GLPbo::Object GLPbo::objs[static_cast<size_t>(GLPbo::ObjectTypes::Total)];
float* GLPbo::depth_buffer = nullptr;

// global variables
namespace
{
	glm::mat4 vp_mtx;
	size_t obj_index;
	bool r_key_pressed, w_key_pressed, m_key_pressed, x_key_pressed, z_key_pressed;
	bool x_key_toggled, z_key_toggled;

	float constexpr rotating_speed = 30.0f;
	size_t constexpr max_rm  = static_cast<size_t>(GLPbo::RenderMode::Total),
					 max_obj = static_cast<size_t>(GLPbo::ObjectTypes::Total);

	glm::mat4 view(1.0f), proj(1.0f), device(1.0f);

	GLPbo::PointLight light{ { 1.0f, 1.0f, 1.0f}, { 0.0f, 0.0f, 10.0f } };
	GLPbo::Texture texture;
	glm::vec3 axis_of_rotation{ 0.0f, 1.0f, 0.0f };	

	// Core 11
	std::string model_names[2] = { "ogre", "cube" };
	std::string texture_file = "ogre.tex";
	glm::vec3 cam_pos{ 0.0f, 0.0f, 10.0f }, target{ 0.0f, 0.0f, 0.0f }, up{ 0.0f, 1.0f, 0.0f };
	float near = 8.0f, far = 12.0f, top = 1.5f, bottom = -1.5f, left, right;
	float constexpr UNIFORM_SCALE = 2.0f;

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
	// clear color buffer and depth buffer
	clear_color_buffer(), clear_depth_buffer();

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_R) && !r_key_pressed)
	{
		objs[obj_index].rotating = !objs[obj_index].rotating;
		r_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_R) && r_key_pressed)
		r_key_pressed = false;

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_W) && !w_key_pressed)
	{
		size_t index = static_cast<size_t>(objs[obj_index].rm); (++index) %= max_rm;
		objs[obj_index].rm = static_cast<RenderMode>(index);
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

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_X) && !x_key_pressed)
	{
		x_key_toggled = !x_key_toggled;
		x_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_X) && x_key_pressed)
		x_key_pressed = false;

	if (GLFW_PRESS == glfwGetKey(GLHelper::ptr_window, GLFW_KEY_Z) && !z_key_pressed)
	{
		z_key_toggled = !z_key_toggled;
		z_key_pressed = true;
	}
	if (GLFW_PRESS != glfwGetKey(GLHelper::ptr_window, GLFW_KEY_Z) && z_key_pressed)
		z_key_pressed = false;

	if (objs[obj_index].rotating)
		objs[obj_index].angle += rotating_speed * static_cast<float>(GLHelper::delta_time);

	if (!z_key_toggled && !x_key_toggled)
		axis_of_rotation = { 0.0f, 1.0f, 0.0f };
	else if (z_key_toggled && x_key_toggled)
		axis_of_rotation = { 1.0f, 1.0f, 1.0f };
	else if (x_key_toggled)
		axis_of_rotation = { 1.0f, 1.0f, 0.0f };
	else if (z_key_toggled)
		axis_of_rotation = { 0.0f, 1.0f, 1.0f };

	viewport_xform();

	switch (objs[obj_index].rm)
	{
		case RenderMode::Wireframe:
		{
			wireframe_mode();
			break;
		}
		case RenderMode::Depth_Buffer:
		case RenderMode::Faceted:	case RenderMode::Shaded:
		case RenderMode::Textured:	case RenderMode::Textured_Faceted:
		case RenderMode::Textured_Shaded:
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

	std::string model = "Model: ", mode = "Mode: ";
	Object const& obj = objs[obj_index];

	switch (obj.rm)
	{
		case RenderMode::Wireframe:
		{
			mode += "Wireframe";
			break;
		}
		case RenderMode::Depth_Buffer:
		{
			mode += "Depth Buffer";
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
		case RenderMode::Textured:
		{
			mode += "Textured";
			break;
		}
		case RenderMode::Textured_Faceted:
		{
			mode += "Textured/Faceted";
			break;
		}
		case RenderMode::Textured_Shaded:
		{
			mode += "Textured/Shaded";
			break;
		}
	}

	switch (static_cast<ObjectTypes>(obj_index))
	{
		case ObjectTypes::Ogre:
		{
			model += model_names[0];
			size_t const index = model.find_first_of(' ');
			model[index + 1] = std::toupper(model_names[0][0]);
			break;
		}
		case ObjectTypes::Cube:
		{
			model += model_names[1];
			size_t const index = model.find_first_of(' ');
			model[index + 1] = std::toupper(model_names[1][0]);
			break;
		}
	}

	oss << std::fixed << std::setprecision(2) << "A2 | Wong Man Cong | " << model << " | " << mode <<
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

	float const AR = static_cast<float>(w / h);
	left = AR * bottom, right = AR * top;

	set_clear_color(0, 0, 0, 255);

	depth_buffer = new float[pixel_cnt];

	glCreateTextures(GL_TEXTURE_2D, 1, &texid);
	glTextureStorage2D(texid, 1, GL_RGBA8, width, height);

	glCreateBuffers(1, &pboid);
	glNamedBufferStorage(pboid, byte_cnt, nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);

	setup_quad_vao();
	setup_shdrpgm();

	obj_index = 0;

	r_key_pressed = w_key_pressed = m_key_pressed = x_key_pressed = z_key_pressed = false;
	x_key_toggled = z_key_toggled = false;

	glEnable(GL_SCISSOR_TEST);
	glScissor(0, 0, width, height);

	viewport_mtx();
	proj = glm::ortho(left, right, bottom, top, near, far), view = view_mtx(cam_pos, target, up);

	device = vp_mtx * proj * view;

	load_scene();
	objs[1].angle = 30.0f;
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
	delete[] depth_buffer;
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
	for (size_t i = 0; i < 2; ++i)
	{
		// loading using my own parser
		if (!my_own_obj_parser("../meshes/" + model_names[i] + ".obj", objs[i].pos, objs[i].nml, objs[i].tex, objs[i].pos_idx, objs[i].nml_idx, objs[i].tex_idx, true, true))
		{
			std::cerr << "Unable to load: " << model_names[i] << ". File is either not present, unreadable or doesn't follow the OBJ file format" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		size_t const pos_size = objs[i].pos.size(), clr_size = objs[i].pos_idx.size() / 3;
		objs[i].pd = std::vector<glm::vec3>(pos_size);
	}

	std::ifstream ifs( "../images/" + texture_file, std::ios::binary );
	if (ifs.fail())
	{
		std::cerr << "Unable to open " << texture_file << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// computing the relevant things to store for texture width, height and image's data
	size_t bytes_per_texel = 0;
	ifs.read((char*)&texture.width  , sizeof(size_t));
	ifs.read((char*)&texture.height , sizeof(size_t));
	ifs.read((char*)&bytes_per_texel, sizeof(size_t));

	size_t const pixel_cnt = texture.width * texture.height;
	texture.texel = std::vector<Color>(pixel_cnt);
	GLubyte r = 0, g = 0, b = 0;
	for (size_t i = 0; i < pixel_cnt; ++i)
	{
		ifs.read((char*)&r, sizeof(GLubyte));
		ifs.read((char*)&g, sizeof(GLubyte));
		ifs.read((char*)&b, sizeof(GLubyte));

		texture.texel[i] = Color{ static_cast<GLubyte>(r), static_cast<GLubyte>(g), static_cast<GLubyte>(b) };;
	}
}

/*  _________________________________________________________________________ */
/*! clear_depth_buffer

@param	none

@return none

clears the depth buffer to store the largest depth value
*/
void GLPbo::clear_depth_buffer()
{
	std::fill_n(depth_buffer, pixel_cnt, 1.0f);
}

/*  _________________________________________________________________________ */
/*! set_pixel_color

@param	x: x coordinate
		y: y coordinate
		clr: clr at (x,y) coordinate

@return none

set a particular color at (x,y) coordinate
*/
void GLPbo::set_pixel_color(int x, int y, Color clr)
{
	if (0 > x || width <= x || 0 > y || height <= y)
		return;
	size_t const index = static_cast<size_t>(y * width + x);
	*(ptr_to_pbo + index) = clr;
}

/*  _________________________________________________________________________ */
/*! set_pixel_color

@param	x: x coordinate
		y: y coordinate
		z: z coordinate
		clr: clr at (x,y) coordinate

@return none

set a particular color at (x,y) coordinate, checks with depth buffer before setting the color
at (x,y) coordinate
*/
void GLPbo::set_pixel_color(int x, int y, float z, Color clr)
{
	if (0 > x || width <= x || 0 > y || height <= y)
		return;
	size_t const index = static_cast<size_t>(y * width + x);
	if (*(depth_buffer + index) > z)
	{
		*(depth_buffer + index) = z;
		set_pixel_color(x, y, clr);
	}
}

/*  _________________________________________________________________________ */
/*! GetTexel

@param	coord: interpolated texture coordinate 

@return Color value at coord based on the texture

Get the texel color based off the interpolated texture coordinates
*/
GLPbo::Color GLPbo::GetTexel(glm::vec2 const& coord)
{
	size_t const s = static_cast<size_t>(std::floor(coord.s * texture.width));
	size_t const t = static_cast<size_t>(std::floor(coord.t * texture.height));

	if (static_cast<size_t>(texture.width) <= s || static_cast<size_t>(texture.height) <= t)
		return { 255, 255, 255, 255 };
	return texture.texel[t * texture.width + s];
}

/*  _________________________________________________________________________ */
/*! viewport_mtx

@param	none

@return none

helper function to generate a viewport mtx based on the window's width and height
*/
void GLPbo::viewport_mtx()
{
	float const half_width = static_cast<float>(GLPbo::width >> 1), half_height = static_cast<float>(GLPbo::height >> 1);
	vp_mtx = glm::mat4
	{
		glm::vec4{ half_width, 0.0f		  , 0.0f, 0.0f },
		glm::vec4{ 0.0f		 , half_height, 0.0f, 0.0f },
		glm::vec4{ 0.0f		 , 0.0f		  , 0.5f, 0.0f },
		glm::vec4{ half_width, half_height, 0.5f, 1.0f },
	};
}

/*  _________________________________________________________________________ */
/*! viewport_xform

@param	none

@return none

transform the ndc coordinates to window coordinate
*/
void GLPbo::viewport_xform()
{
	Object& obj = objs[obj_index];
	size_t pos_size = obj.pos.size();
	// creating alias for obj's pos and pd
	std::vector<glm::vec3> const& pos = obj.pos; std::vector<glm::vec3>& pd = obj.pd;

	// model matrix
	obj.model = model_mtx(obj.angle, axis_of_rotation, glm::vec3(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE));
	glm::mat4 mtx = device * obj.model;

	// looping thru each obj's position and determine their device coordinate
	for (size_t i = 0; i < pos_size; ++i)
		pd[i] = mtx * glm::vec4{ pos[i].x, pos[i].y, pos[i].z, 1.0f };
}

/*  _________________________________________________________________________ */
/*! model_mtx

@param	angle: angle of rotation
		av	 : axis vector to rotate about
		s	 : scale value

@return model matrix based on object's angle, axis of rotation and scale

Does calculation of model matrix immediately and return a model matrix
based on object's angle and scale
*/
glm::mat4 GLPbo::model_mtx(float angle, glm::vec3 const& av, glm::vec3 const& s)
{
	float const rad = glm::radians(angle);
	float const cos = glm::cos(rad), sin = glm::sin(rad), cos_min_one = 1.0f - cos;
	glm::vec3 n = glm::normalize(av);
	float const ni_sq = n.x * n.x * cos_min_one, nj_sq = n.y * n.y * cos_min_one, nk_sq = n.z * n.z * cos_min_one;
	float const ninj  = n.x * n.y * cos_min_one, nink  = n.x * n.z * cos_min_one, njnk  = n.y * n.z * cos_min_one;
	float const ni = n.x * sin, nj = n.y * sin, nk = n.z * sin;

	return glm::mat4
	{
		glm::vec4{ s.x * (cos + ni_sq), s.x * (ninj + nk), s.x * (nink - nj), 0.0f },
		glm::vec4{ s.y * (ninj - nk), s.y * (cos + nj_sq), s.y * (njnk + ni), 0.0f },
		glm::vec4{ s.z * (nink + nj), s.z * (njnk - ni), s.z * (cos + nk_sq), 0.0f },
		glm::vec4{ 0.0f				, 0.0f			   , 0.0f				, 1.0f },
	};
}

glm::mat4 GLPbo::view_mtx(glm::vec3 pos, glm::vec3 tgt, glm::vec3 up)
{
	glm::vec3 const w{ glm::normalize(pos - tgt) };
	glm::vec3 const u{ glm::normalize(glm::cross(up, w)) };
	glm::vec3 const v{ glm::cross(w, u) };

	return glm::mat4
	{
		glm::vec4{ u.x, v.x, w.x, 0.0f },
		glm::vec4{ u.y, v.y, w.y, 0.0f },
		glm::vec4{ u.z, v.z, w.z, 0.0f },
		glm::vec4{ -glm::dot(u, pos), -glm::dot(v, pos), -glm::dot(w, pos), 1.0f },
	};
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
	size_t iterations = objs[obj_index].pos_idx.size() / 3;

	std::vector<unsigned short> const& idx = objs[obj_index].pos_idx;
	Object& obj = objs[obj_index];
	obj.culled = 0;

	for (size_t i = 0; i < iterations; ++i)
	{
		size_t i0 = i * 3, i1 = i * 3 + 1, i2 = i * 3 + 2;
		glm::vec3 pos[3]{};
		pos[0] = obj.pd[ idx[i0] ];
		pos[1] = obj.pd[ idx[i1] ];
		pos[2] = obj.pd[ idx[i2] ];

		if (backface_cull(pos))
		{
			++obj.culled;
			continue;
		}

		Color const& clr = Color(0, 0, 255, 255);

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
float GLPbo::triangle_area(glm::vec3 const pos[3])
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
	glm::vec3 pos[3]{ { x0, y0, 0.0f }, { x1, y1, 0.0f }, { x2, y2, 0.0f } };
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
bool GLPbo::backface_cull(glm::vec3 const pos[3])
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
	size_t iterations = objs[obj_index].pos_idx.size() / 3;

	std::vector<unsigned short> const& pos_idx = objs[obj_index].pos_idx, nml_idx = objs[obj_index].nml_idx;
	Object& obj = objs[obj_index];
	obj.culled = 0;
	/*
		to store the light's model matrix by multiplying the inverse of current's object model matrix
		converting light position from world coordinate to model coordinate
	*/
	glm::vec3 lmc(0.0f);

	using cubyte = const GLubyte;
	switch (obj.rm)
	{
		case RenderMode::Faceted:			case RenderMode::Shaded:
		case RenderMode::Textured_Faceted:  case RenderMode::Textured_Shaded:
		{
			lmc = glm::inverse(obj.model) * glm::vec4(light.position.x, light.position.y, light.position.z, 1.0f);
			break;
		}
	}

	for (size_t i = 0; i < iterations; ++i)
	{
		size_t i0 = i * 3, i1 = i * 3 + 1, i2 = i * 3 + 2;
		glm::vec3 pos[3]{};
		pos[0] = obj.pd[ pos_idx[i0] ];
		pos[1] = obj.pd[ pos_idx[i1] ];
		pos[2] = obj.pd[ pos_idx[i2] ];

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
		glm::vec<2, int> min{ 0,0 }, max{ 0,0 };
		min.x = static_cast<int>( std::floor( std::min( pos[0].x, std::min(pos[1].x, pos[2].x) ) ) );
		min.y = static_cast<int>( std::floor( std::min( pos[0].y, std::min(pos[1].y, pos[2].y) ) ) );
		max.x = static_cast<int>( std::ceil ( std::max( pos[0].x, std::max(pos[1].x, pos[2].x) ) ) );
		max.y = static_cast<int>( std::ceil ( std::max( pos[0].y, std::max(pos[1].y, pos[2].y) ) ) );

		// evaulation of pixel to check if it's part of the triangle
		auto evaluation_value = [](Edge const& e, glm::vec2 const& p)
		{
			return e.a * p.x + e.b * p.y + e.c;
		};

		glm::vec2 const pos0 = glm::vec2(static_cast<float>(min.x) + 0.5f, static_cast<float>(min.y) + 0.5f);
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

		// Calculate color at of normal if rendermode is faceted
		Color clr{ 0, 0, 0, 255 };
		glm::vec3 clr0{ 0.0f, 0.0f, 0.0f }, clr1{ 0.0f, 0.0f, 0.0f }, clr2{ 0.0f, 0.0f, 0.0f };
		float angle = 0.0f, angle0 = 0.0f, angle1 = 0.0f, angle2 = 0.0f;
		switch (obj.rm)
		{
			case RenderMode::Faceted: case RenderMode::Textured_Faceted:
			{
				glm::vec3 const& p0 = obj.pos[ pos_idx[i0] ],
								 p1 = obj.pos[ pos_idx[i1] ],
								 p2 = obj.pos[ pos_idx[i2] ];

				glm::vec3 const l = glm::normalize(lmc - ((p0 + p1 + p2) / 3.0f)); // light position - centroid
				glm::vec3 const n = glm::normalize(glm::cross((p1 - p0), (p2 - p0)));
				angle = std::max(0.0f, glm::dot(n, l));

				cubyte r = static_cast<GLubyte>(light.intensity.r * angle * 255.0f),
					   g = static_cast<GLubyte>(light.intensity.g * angle * 255.0f),
					   b = static_cast<GLubyte>(light.intensity.b * angle * 255.0f);
				clr = Color{ r, g, b, 255 };
				break;
			}
			case RenderMode::Shaded: case RenderMode::Textured_Shaded:
			{
				glm::vec3 const& n0 = obj.nml[ nml_idx[i0] ], n1 = obj.nml[ nml_idx[i1] ], n2 = obj.nml[ nml_idx[i2] ];
				glm::vec3 const& p0 = obj.pos[ pos_idx[i0] ], p1 = obj.pos[ pos_idx[i1] ], p2 = obj.pos[ pos_idx[i2] ];
				glm::vec3 const  l0 = glm::normalize( lmc - p0 ), l1 = glm::normalize( lmc - p1 ), l2 = glm::normalize( lmc - p2 );
				angle0 = std::max(0.0f, glm::dot(n0, l0)), angle1 = std::max(0.0f, glm::dot(n1, l1)), angle2 = std::max(0.0f, glm::dot(n2, l2));

				clr0 = light.intensity * angle0 * 255.0f;
				clr1 = light.intensity * angle1 * 255.0f;
				clr2 = light.intensity * angle2 * 255.0f;
			}
		}

		// Calculate the new area to use for the calculation of color when linearly interpolating thru the vertices
		A = 1.0f / (2.0f * A);
		// Vertical area of triangle
		float vTri0 = vEval0 * A, vTri1 = vEval1 * A, vTri2 = vEval2 * A;
		float const tri0_x_inc = tri.e0.a * A, tri1_x_inc = tri.e1.a * A, tri2_x_inc = tri.e2.a * A;
		float const tri0_y_inc = tri.e0.b * A, tri1_y_inc = tri.e1.b * A, tri2_y_inc = tri.e2.b * A;

		auto TexelColor = [&](float hTri0, float hTri1, float hTri2)
		{
			// texture coordinates
			glm::vec2 const& t0 = obj.tex[ obj.tex_idx[i0] ],
							 t1 = obj.tex[ obj.tex_idx[i1] ],
							 t2 = obj.tex[ obj.tex_idx[i2] ];

			// interpolate texture coordinate
			glm::vec2 const& tex_coord = hTri0 * t0 + hTri1 * t1 + hTri2 * t2;
			return GetTexel(tex_coord);
		};

		for (int y = min.y; y < max.y; ++y)
		{
			float hEval0 = vEval0, hEval1 = vEval1, hEval2 = vEval2;
			float hTri0 = vTri0, hTri1 = vTri1, hTri2 = vTri2;
			for (int x = min.x; x < max.x; ++x)
			{
				if (point_in_triangle(hEval0, hEval1, hEval2))
				{
					// interpolates the depth
					float const z = (hTri0 * obj.pd[ pos_idx[i0] ].z + hTri1 * obj.pd[ pos_idx[i1] ].z + hTri2 * obj.pd[ pos_idx[i2] ].z);

					switch (obj.rm)
					{
						case RenderMode::Depth_Buffer:
						{
							cubyte clr = static_cast<GLubyte>(z * 255.0f);

							set_pixel_color(x, y, z, Color{ clr, clr, clr, 255 });

							break;
						}
						case RenderMode::Faceted:
						{
							set_pixel_color(x, y, z, clr);
							break;
						}
						case RenderMode::Shaded:
						{
							// final color
							glm::vec3 const fColor = (clr0 * hTri0 + clr1 * hTri1 + clr2 * hTri2);
							clr =
							{
								static_cast<GLubyte>(fColor.r),
								static_cast<GLubyte>(fColor.g),
								static_cast<GLubyte>(fColor.b),
								255
							};

							set_pixel_color(x, y, z, clr );

							break;
						}
						case RenderMode::Textured:
						{
							clr = TexelColor(hTri0, hTri1, hTri2);
							set_pixel_color(x, y, z, clr);

							break;
						}
						case RenderMode::Textured_Faceted:
						{
							Color texel = TexelColor(hTri0, hTri1, hTri2);
							glm::vec3 const& defuse
							{
								static_cast<float>(texel.r) / 255.0f,
								static_cast<float>(texel.g) / 255.0f,
								static_cast<float>(texel.b) / 255.0f,
							};

							cubyte r = static_cast<GLubyte>(light.intensity.r * angle * defuse.r * 255.0f),
								   g = static_cast<GLubyte>(light.intensity.g * angle * defuse.g * 255.0f),
								   b = static_cast<GLubyte>(light.intensity.b * angle * defuse.b * 255.0f);
							clr = Color{ r, g, b, 255 };

							set_pixel_color(x, y, z, clr);

							break;
						}
						case RenderMode::Textured_Shaded:
						{
							Color texel = TexelColor(hTri0, hTri1, hTri2);
							glm::vec3 const& defuse
							{
								static_cast<float>(texel.r) / 255.0f,
								static_cast<float>(texel.g) / 255.0f,
								static_cast<float>(texel.b) / 255.0f,
							};

							clr0 = light.intensity * defuse * angle0 * 255.0f;
							clr1 = light.intensity * defuse * angle1 * 255.0f;
							clr2 = light.intensity * defuse * angle2 * 255.0f;

							glm::vec3 const fColor = (clr0 * hTri0 + clr1 * hTri1 + clr2 * hTri2);
							clr =
							{
								static_cast<GLubyte>(fColor.r),
								static_cast<GLubyte>(fColor.g),
								static_cast<GLubyte>(fColor.b),
								255
							};

							set_pixel_color(x, y, z, clr);

							break;
						}
					}
				}
				hEval0 += tri.e0.a, hEval1 += tri.e1.a, hEval2 += tri.e2.a;
				hTri0 += tri0_x_inc, hTri1 += tri1_x_inc, hTri2 += tri2_x_inc;
			}
			vEval0 += tri.e0.b, vEval1 += tri.e1.b, vEval2 += tri.e2.b;
			vTri0 += tri0_y_inc, vTri1 += tri1_y_inc, vTri2 += tri2_y_inc;
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
	std::vector<unsigned short>&  normal_indices,
	std::vector<unsigned short>&  texture_indices,
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

	using u16 = unsigned short;

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
				size_t const first_separator = line.find_first_of('/'), last_separator = line.find_last_of('/');
				position_indices.push_back( static_cast<u16>( std::stoi( line.substr(0, first_separator) ) - 1) );
				texture_indices.push_back ( static_cast<u16>( std::stoi( line.substr(first_separator + 1, last_separator) ) - 1) );
				normal_indices.push_back  ( static_cast<u16>( std::stoi( line.substr(last_separator  + 1, line.size()) ) - 1) );
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
		size_t const nml_size = normals.size();
		for (size_t i = 0; i < nml_size; ++i)
			normals[i] = glm::vec3(0.0f);

		using u16 = unsigned short;
		for (size_t i = 0; i < nml_size; i += 3)
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