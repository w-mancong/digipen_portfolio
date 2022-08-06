/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		02/06/2022

This file implements functions that creates a box shape and another mystery shape
The program starts of with no objects being rendered on the screen.
When the user press the left mouse button, objects will spawn with a maximum number 
of 32'768 objects being rendered on the screen at any one point.
When the program spawns reaches the max, it will start decreasing until it renders
1 object on the screen.
The cycle continues from there.

Press 'P' to change between polygon modes. The three modes are:
1) GL_FILL 2) GL_LINE 3) GL_POINT
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#define _USE_MATH_DEFINES
#include <math.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
double fps						= 0.0;

GLuint constexpr TOTAL_MODES	= 3, TOTAL_OBJECTS = 32'768;
GLenum polygonMode[TOTAL_MODES] = { GL_FILL, GL_LINE, GL_POINT };
GLuint polygonIdx				= 0, head = 0, tail = 0;	// head and tail are indexes used for swap_ranges
size_t live_objects				= 0, box = 0, mystery = 0;
GLboolean keystateFlag			= GL_FALSE, mousestateFlag = GL_FALSE;

std::vector<GLApp::GLModel> GLApp::models;
std::vector<GLSLShader> GLApp::shdrpgms;
std::vector<GLApp::GLObject> GLApp::objects(TOTAL_OBJECTS);

enum class Status
{
	Spawn,
	Despawn
};

Status status;

/*  _________________________________________________________________________ */
/*! Random

@param	min: lower bound
		max: upper bound

@return	a random number

Randomizes a number between min and max
*/
float Random(float min, float max)
{
	std::random_device rd; std::mt19937 gen(rd()); std::uniform_real_distribution<float> dist(min, max);
	return dist(gen);
}

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

/*  _________________________________________________________________________ */
/*! Spawn

@param none

@return none

Spawn/Despawn objects in the scene based on the current status of the program
*/
void Spawn(void)
{
	switch (status)
	{
		case Status::Spawn:
		{
			if (0 != live_objects)
				live_objects <<= 1;
			else
				live_objects = 1;		
			for (size_t i = live_objects >> 1; i < live_objects; ++i)
				GLApp::objects[i].init();
			break;
		}
		case Status::Despawn:
		{
			tail = live_objects, head = (live_objects >>= 1);
			std::swap_ranges(GLApp::objects.begin() + head, GLApp::objects.begin() + tail, GLApp::objects.begin());
			break;
		}
	}
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

	//srand(static_cast<unsigned int>(time(NULL)));

	glViewport(0, 0, GLHelper::width, GLHelper::height);

	init_models_cont();

	vpss shdr_file_names
	{
		std::make_pair<std::string, std::string>("../shaders/my-tutorial-3.vert", "../shaders/my-tutorial-3.frag"),
	};
	init_shdrpgms_cont(shdr_file_names);

	glEnable(GL_PRIMITIVE_RESTART);
	GLushort const index = static_cast<GLushort>(GL_PRIMITIVE_RESTART_INDEX);
	glPrimitiveRestartIndex(index);
}

/*  _________________________________________________________________________ */
/*! update

@param none

@return none

Spawn/Despawn objects onto the screen whenever the left mouse button is pressed
Update each object's model_to_ndc matrix
*/
void GLApp::update() 
{
	if (GLHelper::mousestateLeft && !mousestateFlag)
	{
		// init things on screen
		if (1 >= live_objects)
			status = Status::Spawn;
		else if (32'768 <= live_objects)
			status = Status::Despawn;
		Spawn();
		mousestateFlag = GL_TRUE;
	}
	else if (!GLHelper::mousestateLeft && mousestateFlag)
		mousestateFlag = GL_FALSE;

	box = 0, mystery = 0;
	for (size_t i = 0; i < live_objects; ++i)
	{
		objects[i].update(GLHelper::delta_time);
		if (objects[i].mdl_ref) ++mystery;
		else ++box;
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

	std::string mode;
	switch (polygonMode[polygonIdx])
	{
		case GL_FILL:
		{
			mode = "Fill";
			break;
		}
		case GL_LINE:
		{
			mode = "Line";
			break;
		}
		case GL_POINT:
		{
			mode = "Point";
			break;
		}
	}
	oss << std::fixed << std::setprecision(2) << "Tutorial 3 | Wong Man Cong | "
		"Mode: " << mode <<
		" | Obj: " << live_objects <<
		" | Box: " << box <<
		" | Mystery: " << mystery <<
		" | " << fps;
	glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());

	if (GLHelper::keystateP && !keystateFlag)
	{
		(++polygonIdx) %= TOTAL_MODES;
		keystateFlag = GL_TRUE;
	}
	else if(!GLHelper::keystateP && keystateFlag)
		keystateFlag = GL_FALSE;

	glPolygonMode(GL_FRONT_AND_BACK, polygonMode[polygonIdx]);
	switch (polygonMode[polygonIdx])
	{
		case GL_LINE:
		{
			glLineWidth(3.f);
			break;
		}
		case GL_POINT:
		{
			glPointSize(10.0f);
			break;
		}
	}
	for (size_t i = 0; i < live_objects; ++i)
		objects[i].draw();
	glPointSize(1.0f); glLineWidth(1.0f);
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
/*! init

@param none

@return none

Geometry instancing of object by randomizing the position, speed of rotation,
initial angle of rotation, scale and the object
*/
void GLApp::GLObject::init(void)
{
	angle_speed = Random(-30.0f, 30.0f), angle_disp = Random(-360.0f, 360.0f);
	position = glm::vec2( Random(-1.0f, 1.0f) * (GLHelper::width >> 1), Random(-1.0f, 1.0f) * (GLHelper::height >> 1) );
	scale = glm::vec2(Random(50.0f, 100.0f), Random(50.0f, 100.0f));
	mdl_ref = Random(0, 1); shdr_ref = 0;
}

/*  _________________________________________________________________________ */
/*! update

@param	dt: delta time of the program

@return none

update function of each object. Calculates the model matrix
*/
void GLApp::GLObject::update(GLdouble dt)
{
	angle_disp += angle_speed * static_cast<GLfloat>(dt);
	model = glm::mat3(1.0f);
	float const rad = glm::radians(angle_disp), cos = std::cosf(rad), sin = std::sinf(rad);
	float const nx  = 1.0f / static_cast<float>(GLHelper::width >> 1), ny = 1.0f / static_cast<float>(GLHelper::height >> 1);

	model[0][0] =   scale.x * nx * cos;		model[0][1] = scale.x * ny * sin;
	model[1][0] = -(scale.y * nx * sin);	model[1][1] = scale.y * ny * cos;
	model[2][0] = nx * position.x;			model[2][1] = ny * position.y;
}

/*  _________________________________________________________________________ */
/*! draw

@param none

@return none

Drawing of individual object based on their model matrix
*/
void GLApp::GLObject::draw(void) const
{
	// use shader program
	glUseProgram(shdrpgms[shdr_ref].GetHandle());
	GLint uniform_var_loc1 = glGetUniformLocation(shdrpgms[shdr_ref].GetHandle(), "uModel_to_NDC");
	if (0 <= uniform_var_loc1)
		glUniformMatrix3fv(uniform_var_loc1, 1, GL_FALSE, glm::value_ptr(model));
	else
	{
		std::cerr << "Uniform variable doesn't exist!!!" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	// bind vao
	glBindVertexArray(models[mdl_ref].vaoid);
	glDrawElements(models[mdl_ref].primitive_type, models[mdl_ref].draw_cnt, GL_UNSIGNED_SHORT, nullptr);

	// Unbind and stop using vao and shader program
	glBindVertexArray(0);
	glUseProgram(0);
}

/*  _________________________________________________________________________ */
/*! init_shdrpgms_cont

@param	shdr: file paths to both vertex and fragment shader for each individual program

@return none

initializes shader instancing based on their filepath
*/
void GLApp::init_shdrpgms_cont(vpss const& shdr)
{
	for (auto const& x : shdr)
	{
		std::vector<std::pair<GLenum, std::string>> shdr_files;
		shdr_files.emplace_back(std::make_pair(GL_VERTEX_SHADER, x.first));
		shdr_files.emplace_back(std::make_pair(GL_FRAGMENT_SHADER, x.second));

		GLSLShader pgm;
		pgm.CompileLinkValidate(shdr_files);
		// insert shader program into container
		shdrpgms.emplace_back(pgm);
	}
}

/*  _________________________________________________________________________ */
/*! init_models_cont

@param none

@return none

models vector containing a copy of box_model and mystery_model
*/
void GLApp::init_models_cont(void)
{
	models.emplace_back(GLApp::box_model());
	models.emplace_back(GLApp::mystery_model());
}

/*  _________________________________________________________________________ */
/*! box_model

@param none

@return none

sending the vertex attribute of a box model to the gpu
*/
GLApp::GLModel GLApp::box_model(void)
{
	std::vector<glm::vec2> pos_vtx{ 4 };	// box have 4 vertices
	// 0: top left, 1: btm left, 2: btm right, 3: top right
	pos_vtx[0] = glm::vec2(-0.5f, 0.5f); pos_vtx[1] = glm::vec2(-0.5f, -0.5f);
	pos_vtx[2] = glm::vec2(0.5f, -0.5f); pos_vtx[3] = glm::vec2(0.5f, 0.5f);
	std::vector<glm::vec3> clr_vtx{ 4 };

	for (size_t i = 0; i < 4; ++i)
	{
		float const r = Random(0.0f, 1.0f), g = Random(0.0f, 1.0f), b = Random(0.0f, 1.0f);
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vao, vbo, ebo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	glCreateVertexArrays(1, &vao);
	// position attribute
	glEnableVertexArrayAttrib(vao, 0);
	glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(vao, 1);
	glVertexArrayVertexBuffer(vao, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 1, 1);

	std::vector<GLushort> idx_vtx{ 0, 1, 3, 2 };
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLushort) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vao, ebo);

	glBindVertexArray(0);
	return GLModel { GL_TRIANGLE_STRIP, idx_vtx.size(), vao, idx_vtx.size() };
}

/*  _________________________________________________________________________ */
/*! mystery_model

@param none

@return none

sending the vertex attribute of a mystery_model to the gpu
*/
GLApp::GLModel GLApp::mystery_model(void)
{
	std::vector<glm::vec2> pos_vtx{ 6 }; // mystery shape has 6 vertex attributes
	pos_vtx[0] = glm::vec2(0.0f, 0.8f);   pos_vtx[1] = glm::vec2(-0.2f, 0.1f); pos_vtx[2] = glm::vec2(0.2f, 0.1f);
	pos_vtx[3] = glm::vec2(-0.8f, -0.4f); pos_vtx[4] = glm::vec2(0.0f, -0.1f); pos_vtx[5] = glm::vec2(0.8f, -0.4f);

	std::vector<glm::vec3> clr_vtx{ 6 };

	for (size_t i = 0; i < 6; ++i)
	{
		float const r = Random(0.0f, 1.0f), g = Random(0.0f, 1.0f), b = Random(0.0f, 1.0f);
		clr_vtx[i] = glm::vec3{ r, g, b };
	}

	GLuint vao, vbo, ebo;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	glCreateVertexArrays(1, &vao);
	// position attribute
	glEnableVertexArrayAttrib(vao, 0);
	glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 0, 0);

	// color attribute
	glEnableVertexArrayAttrib(vao, 1);
	glVertexArrayVertexBuffer(vao, 1, vbo, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 1, 1);

	std::vector<GLushort> idx_vtx{ 0, 1, 2, 4, 5, GL_PRIMITIVE_RESTART_INDEX, 1, 4, 3 };
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLushort) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vao, ebo);

	glBindVertexArray(0);
	return GLModel{ GL_TRIANGLE_STRIP, idx_vtx.size(), vao, idx_vtx.size() };
}