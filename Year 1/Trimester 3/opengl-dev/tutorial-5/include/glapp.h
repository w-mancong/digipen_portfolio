/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		16/06/2022

This file implement function declarations to load textures, shader and a rectangle
model to be rendered onto the screen
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
  /*! update_inputs

  @param	none

  @return none

  check the inputs by the user and update the keys accordingly
  The type of keys and it's functionalities are explained at the top
  */
  static void update_inputs();

  /*  _________________________________________________________________________ */
  /*! update_time

  @param	none

  @return none

  linearly interpolate uSize between MIN_SIZE and MAX_SIZE over a period of TOTAL_TIME
  */
  static void update_time();

  /*  _________________________________________________________________________ */
  /*! init_shader

  @param	vtx_shdr: file path to vertex shader
		  fgm_shdr: file path to fragment shader

  @return none

  initialise the shader program that gonna be used
  */
  static void init_shader(std::string const& vtx_shdr, std::string const& fgm_shdr);

  /*  _________________________________________________________________________ */
  /*!	init_rect_model

  @param	none

  @return none

  initializes a model of a rectangle that covers the entire viewport using the
  Array Of Structures format (AOS)
  */
  static void init_rect_model();

  /*  _________________________________________________________________________ */
  /*! draw_rect_model

  @param	none

  @return none

  draws the rectangle model
  */
  static void draw_rect_model();

  /*  _________________________________________________________________________ */
  /*! setup_texobj

  @param	file_path: path to the texture file

  @return a handle to the texture object created

  Read and create a texture object handle
  */
  static GLuint setup_texobj(std::string const& file_path);

  // encapsulates state required to render a geometrical model
  struct GLModel
  {
	  GLenum		primitive_type{ 0 };	// which OpenGL primitive to be rendered
	  GLuint		vaoid{ 0 };				// handle to VAO
	  GLuint		vbo{ 0 };
	  GLuint		draw_cnt{ 0 };
  };

  static GLModel rectModel;
  static GLSLShader shdrpgm;
  static GLuint duck_texture;
};

#endif /* GLAPP_H */
