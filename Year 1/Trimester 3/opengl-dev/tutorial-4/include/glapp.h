/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		09/06/2022

This file implements function declaration to init scene and draw objects onto screen
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */
#ifndef GLAPP_H
#define GLAPP_H

#include <glhelper.h>
#include <glslshader.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <sstream>
#include <iomanip>
#include <random>
#include <map>

/*                                                                   includes
----------------------------------------------------------------------------- */

struct GLApp 
{
  /*  _________________________________________________________________________ */
  /*! init
  
  @param none
  
  @return none
  
  Initializes viewport, vao and shader program
  */
  static void init();

  /*  _________________________________________________________________________ */
  /*! update

  @param none

  @return none

  Spawn/Despawn objects onto the screen whenever the left mouse button is pressed
  Update each object's model_to_ndc matrix
  */
  static void update();

  /*  _________________________________________________________________________ */
  /*! draw

  @param none

  @return none

  clear current framebuffer and call the draw call to draw to previous framebuffer
  */
  static void draw();

  /*  _________________________________________________________________________ */
  /*! cleanup

  @param none

  @return none

  empty for now
  */
  static void cleanup();

  /*  _________________________________________________________________________ */
  /*! insert_shdrpgm

  @param	pgm_name: name of the shader program
		  vtx_shdr: file path to vertex shader
		  fgm_shdr: file path to fragment shader

  @return none

  initializes shader instancing based on their filepath
  */
  static void insert_shdrpgm(std::string const& pgm_name, std::string const& vtx_shdr, std::string const& fgm_shdr);

  /*  _________________________________________________________________________ */
  /*! init_scene

  @param	scene_file: file path to the scene to be loaded

  @return none

  Initializes objects in the scene that will be displayed on the screen
  */
  static void init_scene(std::string const& scene_file);

  /*  _________________________________________________________________________ */
  /*! init_model

  @param	model_file: name of the model to be loaded

  @return none

  Initializes an instance of the model
  */
  static void init_model(std::string const& model_file);

  // encapsulates state required to render a geometrical model
  struct GLModel
  {
	  GLenum		primitive_type;	// which OpenGL primitive to be rendered
	  GLuint		primitive_cnt;
	  GLuint		vaoid;			// handle to VAO
	  GLuint		draw_cnt;
  };

  struct GLObject
  {
	  glm::vec2 position, scale, orientation; // orientation.x => angle_disp, orientation.y => angle_speed
	  glm::vec3 color;
	  glm::mat3 mdl_xform, mdl_to_ndc_xform;
	  std::map<std::string, GLModel>::iterator mdl_ref;
	  std::map<std::string, GLSLShader>::iterator shdr_ref;

	  /*  _________________________________________________________________________ */
	  /*! update

	  @param  none

	  @return none

	  update function of each object. Calculates the model matrix
	  */
	  void update();

	  /*  _________________________________________________________________________ */
	  /*! draw

	  @param none

	  @return none

	  Drawing of individual object based on their model matrix
	  */
	  void draw(void) const;
  };

  struct Camera2D
  {
	  GLObject* pgo{ nullptr }; // pointer to game object that embeds camera
	  glm::vec2 right, up;
	  glm::mat3 view_xform{ glm::mat3(1.0f) }, camwin_to_ndc_xform{ glm::mat3(1.0f) }, world_to_ndc_xform{ glm::mat3(1.0f) };
	  GLint height{ 100 };
	  GLfloat ar{ 0.0f };

	  // window change parameters
	  GLint const min_height{ 500 }, max_height{ 2000 };
	  // height is increasing if 1, decreasing if -1
	  GLint height_dir{ 1 };
	  // increments by which window height is changed per z key press
	  GLint height_chg_val{ 5 };

	  // camera's speed when button U is pressed
	  GLfloat cam_speed{ 2.0f };

	  /*
		v: toggle between freeand first person camera
		z: zoom in and out
	  */ 
	  GLboolean v_flag{ GL_FALSE };

	  /*  _________________________________________________________________________ */
	  /*! init

	  @param	window: pointer to GLFWwindow to calculate the aspect ratio
			  go: pointer to GLObject that the camera is referencing

	  @return none

	  Initializes the static camera in the scene
	  */
	  void init(GLFWwindow* window, GLObject* go);

	  /*  _________________________________________________________________________ */
	  /*! update

	  @param	window: pointer to GLFWwindow to calculate the aspect ratio of camera

	  @return none

	  Update camera based on user's input
	  */
	  void update(GLFWwindow* window);
  };

  // data member to represent geometric model to be rendered
  // C++ requires this object to have a definition in glapp.cpp!!!
  static std::map<std::string, GLModel> models;
  static std::map<std::string, GLSLShader> shdrpgms;
  static std::map<std::string, GLObject> objects;
  static Camera2D camera;
};

#endif /* GLAPP_H */
