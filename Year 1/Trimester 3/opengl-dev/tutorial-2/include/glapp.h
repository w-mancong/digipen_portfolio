/*!
@file		glapp.cpp
@author		pghali@digipen.edu
@co-author	w.mancong@digipen.edu
@date		26/05/2022

This files implement functions that splits the viewport into 4 different regions
while rendering with different primitive types in each quadrant. The primitive
types used to render each quadrants are: GL_POINTS, GL_LINES, GL_TRIANGLE_FAN and GL_TRIANGLE_STRIP
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */
#ifndef GLAPP_H
#define GLAPP_H

#include <glhelper.h>
#include <glslshader.h>
#include <glm/glm.hpp>
#include <iostream>
#include <algorithm>
#include <array>
#include <sstream>
#include <iomanip>

/*                                                                   includes
----------------------------------------------------------------------------- */

struct GLApp 
{
  static void init();
  static void update();
  static void draw();
  static void cleanup();

  // encapsulates state required to render a geometrical model
  struct GLModel
  {
	  GLenum		primitive_type{ 0 };	// which OpenGL primitive to be rendered
	  GLuint		primitive_cnt{ 0 };
	  GLuint		vaoid{ 0 };			// handle to VAO
	  GLuint		draw_cnt{ 0 };

	  GLSLShader	shdr_pgm;		// which shader program to use

	  void setup_shdrpgm(std::string const& vtx_shdr, std::string const& frg_shdr);
	  void draw();
  };

  /*  _________________________________________________________________________ */
  /*! cleanup

  @param	slices : number of points to be generated horizontally
		  stacks : number of points to be generated vertically
		  vtx_shdr: file path to vertex shader
		  frg_shdr: file path to fragment shader

  Send vertex attribute data that forms points on the screen from cpu to the gpu and binds the buffer object to vaoid
  */
  static GLModel points_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr);

  /*  _________________________________________________________________________ */
  /*! lines_model

  @param	slices : number of lines to be generated horizontally
		  stacks : number of lines to be generated vertically
		  vtx_shdr: file path to vertex shader
		  frg_shdr: file path to fragment shader

  Send vertex attribute information to form a line from cpu to the gpu and binds the buffer object to vaoid
  */
  static GLModel lines_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr);

  /*  _________________________________________________________________________ */
  /*! trifans_model

  @param	slices : total number of slices to form a circle
		  vtx_shdr: file path to vertex shader
		  frg_shdr: file path to fragment shader

  Send vertex attribute information to form a circle from cpu to the gpu and binds the buffer object to vaoid
  */
  static GLModel trifans_model(GLint slices, std::string const& vtx_shdr, std::string const& frg_shdr);

  /*  _________________________________________________________________________ */
  /*! tristrip_model

  @param	slices : number of points to be generated horizontally
		  stacks : number of points to be generated vertically
		  vtx_shdr: file path to vertex shader
		  frg_shdr: file path to fragment shader

  Send vertex attribute information to form a line from cpu to the gpu and binds the buffer object to vaoid
  */
  static GLModel tristrip_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frg_shdr);

  // encapsulate viewport dimensions
  struct GLViewport
  {
	  GLint x, y;
	  GLsizei width, height;
  };
  static std::vector<GLViewport> vps;

  // data member to represent geometric model to be rendered
  // C++ requires this object to have a definition in glapp.cpp!!!
  static std::vector<GLModel> models;
};

#endif /* GLAPP_H */
