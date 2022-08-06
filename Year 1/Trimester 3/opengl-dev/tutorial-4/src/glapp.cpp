/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		09/06/2022

This file implements method to load models' information, objects' in the scene
and shader program to be displayed onto the screen. 

When running the program, user can use 'U', 'H' and 'K' to navigate around the scene.
'Z' key is used to zoom in/out of the scene
'V' key is used to change from first person perspective to a free camera perspective
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#define _USE_MATH_DEFINES
#include <math.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
double fps						= 0.0;

std::map<std::string, GLApp::GLModel> GLApp::models;
std::map<std::string, GLSLShader> GLApp::shdrpgms;
std::map<std::string, GLApp::GLObject> GLApp::objects;
GLApp::Camera2D GLApp::camera;

enum class CameraMode
{
	Free_Camera,
	First_Person,
	Total,
};

CameraMode cameraMode = CameraMode::Free_Camera;
GLshort camIdx = static_cast<GLshort>(cameraMode);

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

	init_scene("../scenes/tutorial-4.scn");

	camera.init(GLHelper::ptr_window, &objects.at("Camera"));
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
	camera.update(GLHelper::ptr_window);

	for (auto it = objects.begin(); it != objects.end(); ++it)
	{
		if (it->first == "Camera")
			continue;
		it->second.update();
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

	oss << std::fixed << std::setprecision(2) << "Tutorial 4 | Wong Man Cong | "
		"Camera Position (" << camera.pgo->position.x << ", " << camera.pgo->position.y <<
		" | Orientation: " << camera.pgo->orientation.x << " degrees" <<
		" | Window height: " << camera.height <<
		" | FPS: " << fps;

	glfwSetWindowTitle(GLHelper::ptr_window, oss.str().c_str());

	for (auto it = objects.begin(); it != objects.end(); ++it)
	{
		if (it->first == "Camera")
			continue;
		it->second.draw();
	}
	camera.pgo->draw();
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
/*! insert_shdrpgm

@param	pgm_name: name of the shader program
		vtx_shdr: file path to vertex shader
		fgm_shdr: file path to fragment shader

@return none

initializes shader instancing based on their filepath
*/
void GLApp::insert_shdrpgm(std::string const& pgm_name, std::string const & vtx_shdr, std::string const& fgm_shdr)
{
	std::vector<std::pair<GLenum, std::string>> shdr_files
	{
		std::make_pair(GL_VERTEX_SHADER, vtx_shdr),
		std::make_pair(GL_FRAGMENT_SHADER, fgm_shdr)
	};
	GLSLShader shdr_pgm;
	shdr_pgm.CompileLinkValidate(shdr_files);
	if (GL_FALSE == shdr_pgm.IsLinked())
	{
		std::cerr << "Unable to compile/link/validate shader programs" << std::endl;
		std::cerr << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	shdrpgms[pgm_name] = shdr_pgm;
}

/*  _________________________________________________________________________ */
/*! init_scene

@param	scene_file: file path to the scene to be loaded

@return none

Initializes objects in the scene that will be displayed on the screen
*/
void GLApp::init_scene(std::string const& scene_file)
{
	std::ifstream ifs{ scene_file, std::ios::in };
	if (!ifs)
	{
		std::cerr << "ERROR: Unable to open scene file: "
			<< scene_file << std::endl;
		std::exit(EXIT_FAILURE);
	}
	ifs.seekg(0, std::ios::beg);

	std::string line;
	std::getline(ifs, line);
	std::istringstream iss{ line };
	int obj_cnt = 0;
	iss >> obj_cnt;
	while (obj_cnt--)
	{
		std::string model_name, object_name, shdrpgm_name, vtx_shdr, fgm_shdr;
		float x		= 0.0f, y		= 0.0f;				// position
		float sx	= 0.0f, sy		= 0.0f;				// scale
		float disp	= 0.0f, speed	= 0.0f;				// initial rotation, speed of rotation
		float r		= 0.0f, g		= 0.0f, b = 0.0f;	// color

		// getting model's name
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> model_name;
		// check if the model have been instantiated
		if (models.end() == models.find(model_name))
			init_model(model_name);

		// object's name
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> object_name;

		// shader program's name
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> shdrpgm_name >> vtx_shdr >> fgm_shdr;
		if (shdrpgms.end() == shdrpgms.find(shdrpgm_name))
			insert_shdrpgm(shdrpgm_name, vtx_shdr, fgm_shdr);

		// color
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> r >> g >> b;

		// scale
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> sx >> sy;

		// orientation (initial angle displacement, speed of rotation)
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> disp >> speed;

		// position
		std::getline(ifs, line);
		iss = std::move(std::istringstream(line));
		iss >> x >> y;

		GLObject obj;
		obj.position			= glm::vec2(x, y);
		obj.scale				= glm::vec2(sx, sy);
		obj.orientation			= glm::vec2(disp, speed);
		obj.color				= glm::vec3(r, g, b);
		obj.mdl_ref				= models.find(model_name);
		obj.shdr_ref			= shdrpgms.find(shdrpgm_name);
		objects[object_name]	= obj;
	}
}

/*  _________________________________________________________________________ */
/*! init_model

@param	model_file: name of the model to be loaded

@return none

Initializes an instance of the model
*/
void GLApp::init_model(std::string const& model_file)
{
	std::ifstream ifs{ "../meshes/" + model_file + ".msh", std::ios::in };
	std::string line, model_name;
	std::vector<glm::vec2> pos_vtx;
	std::vector<GLushort> idx_vtx;
	bool fan_activated = false;

	while (std::getline(ifs, line))
	{
		char prefix = '\0'; // starting character in file
		std::istringstream iss{ line };
		iss >> prefix;
		prefix = std::tolower(prefix);

		// lambda function to add index into vector
		auto push_back = [&](void) {
			GLushort idx = 0;
			iss >> idx;
			idx_vtx.push_back(idx);
		};

		switch (prefix)
		{
			case 'n':
			{
				iss >> model_name;
				break;
			}
			case 'v':
			{
				float x = 0.0f, y = 0.0f;
				iss >> x >> y;
				pos_vtx.push_back( glm::vec2(x, y) );
				break;
			}
			case 't':
			{
				for (size_t i = 0; i < 3; ++i)
					push_back();
				break;
			}
			case 'f':
			{
				if (fan_activated)
					push_back();
				else
				{
					for (size_t i = 0; i < 3; ++i)
						push_back();
					fan_activated = true;
				}
				break;
			}
			default:
				break;
		}
	}

	GLuint vao = 0, vbo = 0, ebo = 0;
	glCreateBuffers(1, &vbo);
	glNamedBufferStorage(vbo, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data(), GL_DYNAMIC_STORAGE_BIT);

	glCreateVertexArrays(1, &vao);
	// position attribute
	glEnableVertexArrayAttrib(vao, 0);
	glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vao, 0, 0);

	// ebo
	glCreateBuffers(1, &ebo);
	glNamedBufferStorage(ebo, sizeof(GLushort) * idx_vtx.size(), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vao, ebo);

	// unbinding process
	glBindVertexArray(0);
	GLenum primitive_type = fan_activated ? GL_TRIANGLE_FAN : GL_TRIANGLES;

	models[model_name] = GLModel{ primitive_type, idx_vtx.size(), vao, idx_vtx.size() };
}

/*  _________________________________________________________________________ */
/*! update

@param	dt: delta time of the program

@return none

update function of each object. Calculates the model matrix
*/
void GLApp::GLObject::update()
{
	orientation.x += orientation.y * static_cast<GLfloat>(GLHelper::delta_time);
	mdl_xform = glm::mat3(1.0f);
	float const rad = glm::radians(orientation.x), cos = std::cosf(rad), sin = std::sinf(rad);

	mdl_xform[0][0] = scale.x * cos;		mdl_xform[0][1] = scale.x * sin;
	mdl_xform[1][0] = -(scale.y * sin);		mdl_xform[1][1] = scale.y * cos;
	mdl_xform[2][0] = position.x;			mdl_xform[2][1] = position.y;

	mdl_to_ndc_xform = glm::mat3(1.0f);
	mdl_to_ndc_xform = camera.world_to_ndc_xform * mdl_xform;
}

/*  _________________________________________________________________________ */
/*! draw

@param none

@return none

Drawing of individual object based on their model matrix
*/
void GLApp::GLObject::draw(void) const
{
	GLSLShader& shdr = shdrpgms.at(shdr_ref->first);
	GLModel const& mdl = models.at(mdl_ref->first);

	shdr.Use();
	glBindVertexArray(mdl.vaoid);
	shdr.SetUniform("uColor", color);
	shdr.SetUniform("uModel_to_NDC", mdl_to_ndc_xform);
	glDrawElements(mdl.primitive_type, mdl.draw_cnt, GL_UNSIGNED_SHORT, nullptr);
	glBindVertexArray(0);
	shdr.UnUse();
}

/*  _________________________________________________________________________ */
/*! init

@param	window: pointer to GLFWwindow to calculate the aspect ratio
		go: pointer to GLObject that the camera is referencing

@return none

Initializes the static camera in the scene
*/
void GLApp::Camera2D::init(GLFWwindow* window, GLObject* go)
{
	pgo = go;

	GLsizei fb_width = 0, fb_height = 0;
	glfwGetFramebufferSize(window, &fb_width, &fb_height);
	ar = static_cast<GLfloat>(fb_width / fb_height);
	height = fb_height;
	
	float const rad = glm::radians(pgo->orientation.x);
	float const cos = std::cosf(rad), sin = std::sinf(rad);
	up		= glm::vec2(-sin, cos);
	right	= glm::vec2(cos, sin);

	view_xform = glm::mat3(1.0f);
	view_xform[2][0] = -(pgo->position.x); view_xform[2][1] = -(pgo->position.y);

	float const w = ar * static_cast<float>(fb_height), h = static_cast<float>(fb_height);
	camwin_to_ndc_xform = glm::mat3(1.0f);
	camwin_to_ndc_xform[0][0] = 2.0f / w;
	camwin_to_ndc_xform[1][1] = 2.0f / h;

	world_to_ndc_xform = glm::mat3(1.0f);
	world_to_ndc_xform = camwin_to_ndc_xform * view_xform;
}

/*  _________________________________________________________________________ */
/*! update

@param	window: pointer to GLFWwindow to calculate the aspect ratio of camera

@return none

Update camera based on user's input 
*/
void GLApp::Camera2D::update(GLFWwindow* window)
{
	GLsizei fb_width = 0, fb_height = 0;
	glfwGetFramebufferSize(window, &fb_width, &fb_height);
	ar = static_cast<GLfloat>(fb_width / fb_height);

	auto update_vector = [&]
	{
		float const rad = glm::radians(pgo->orientation.x);
		float const cos = std::cosf(rad), sin = std::sinf(rad);
		up				= glm::vec2(-sin, cos);
		right			= glm::vec2( cos, sin);
	};

	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_H))
	{
		pgo->orientation.x += pgo->orientation.y;
		if (360.0f < pgo->orientation.x)
			pgo->orientation.x = 0.0f;
		update_vector();
	}
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_K))
	{
		pgo->orientation.x -= pgo->orientation.y;
		if (0.0f > pgo->orientation.x)
			pgo->orientation.x = 360.0f;
		update_vector();
	}
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_U))
	{
		pgo->position += cam_speed * up/* * static_cast<float>(GLHelper::delta_time)*/;
	}
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_Z))
	{
		height += height_chg_val * height_dir;
		if (min_height > height)
			height_dir = 1;
		else if (max_height < height)
			height_dir = -1;
	}
	if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_V) && !v_flag)
	{
		v_flag = GL_TRUE;
		(++camIdx) %= static_cast<GLshort>(CameraMode::Total);
		cameraMode = static_cast<CameraMode>(camIdx);
	}

	if (GLFW_PRESS != glfwGetKey(window, GLFW_KEY_V) && v_flag)
		v_flag = GL_FALSE;

	// updating view matrix
	switch (cameraMode)
	{
		case CameraMode::Free_Camera:
		{
			view_xform = glm::mat3(1.0f);
			view_xform[2][0] = -(pgo->position.x);
			view_xform[2][1] = -(pgo->position.y);
			break;
		}
		case CameraMode::First_Person:
		{
			view_xform = glm::mat3(1.0f);
			float const right_dot_pos	= -(glm::dot(right, pgo->position));
			float const up_dot_pos		= -(glm::dot(   up, pgo->position));

			view_xform[0][0] = right.x;			view_xform[0][1] = up.x;
			view_xform[1][0] = right.y;			view_xform[1][1] = up.y;
			view_xform[2][0] = right_dot_pos;	view_xform[2][1] = up_dot_pos;
			break;
		}
		default:
			break;
	}

	// window-to-ndc matrix
	float const w = ar * static_cast<float>(height), h = static_cast<float>(height);
	camwin_to_ndc_xform = glm::mat3(1.0f);
	camwin_to_ndc_xform[0][0] = 2.0f / w;
	camwin_to_ndc_xform[1][1] = 2.0f / h;

	// world-to-ndc matrix
	world_to_ndc_xform = glm::mat3(1.0f);
	world_to_ndc_xform = camwin_to_ndc_xform * view_xform;

	pgo->mdl_xform = glm::mat3(1.0f);
	float const rad = glm::radians(pgo->orientation.x), cos = std::cosf(rad), sin = std::sinf(rad);

	pgo->mdl_xform[0][0] =  pgo->scale.x * cos;	  pgo->mdl_xform[0][1] = pgo->scale.x * sin;
	pgo->mdl_xform[1][0] = -(pgo->scale.y * sin); pgo->mdl_xform[1][1] = pgo->scale.y * cos;
	pgo->mdl_xform[2][0] =  pgo->position.x;	  pgo->mdl_xform[2][1] = pgo->position.y;

	pgo->mdl_to_ndc_xform = glm::mat3(1.0f);
	pgo->mdl_to_ndc_xform = camera.world_to_ndc_xform * pgo->mdl_xform;
}
