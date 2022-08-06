/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		02/06/2022

This file implements functions declarations to creating, drawing, updating of
a box model and mystery model
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
#include <algorithm>
#include <array>
#include <sstream>
#include <iomanip>
#include <random>

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
	  GLfloat angle_speed, angle_disp;
	  glm::vec2 position, scale;
	  glm::mat3 model;
	  GLuint mdl_ref, shdr_ref;

	  /*  _________________________________________________________________________ */
	  /*! init

	  @param none

	  @return none

	  Geometry instancing of object by randomizing the position, speed of rotation,
	  initial angle of rotation, scale and the object
	  */
	  void init(void);

	  /*  _________________________________________________________________________ */
	  /*! update

	  @param	dt: delta time of the program

	  @return none

	  update function of each object. Calculates the model matrix
	  */
	  void update(GLdouble dt);

	  /*  _________________________________________________________________________ */
	  /*! draw

	  @param none

	  @return none

	  Drawing of individual object based on their model matrix
	  */

	  void draw(void) const;
  };

  using vpss = std::vector<std::pair<std::string, std::string>>;

  /*  _________________________________________________________________________ */
  /*! init_shdrpgms_cont

  @param	shdr: file paths to both vertex and fragment shader for each individual program

  @return none

  initializes shader instancing based on their filepath
  */
  static void init_shdrpgms_cont(vpss const& shdr);

  /*  _________________________________________________________________________ */
  /*! init_models_cont

  @param none

  @return none

  models vector containing a copy of box_model and mystery_model
  */
  static void init_models_cont(void);

  /*  _________________________________________________________________________ */
  /*! box_model

  @param none

  @return none

  sending the vertex attribute of a box model to the gpu
  */
  static GLModel box_model(void);

  /*  _________________________________________________________________________ */
  /*! mystery_model

  @param none

  @return none

  sending the vertex attribute of a mystery_model to the gpu
  */
  static GLModel mystery_model(void);

  // data member to represent geometric model to be rendered
  // C++ requires this object to have a definition in glapp.cpp!!!
  static std::vector<GLModel> models;
  static std::vector<GLSLShader> shdrpgms;
  static std::vector<GLObject> objects;
};

#endif /* GLAPP_H */
