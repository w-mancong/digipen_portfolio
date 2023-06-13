////////////////////////////////////////////////////////////////////////
// A small library of object shapes (ground plane, sphere, and the
// famous Utah teapot), each created as a Vertex Array Object (VAO).
// This is the most efficient way to get geometry into the OpenGL
// graphics pipeline.
//
// Each vertex is specified as four attributes which are made
// available in a vertex shader in the following attribute slots.
//
// position,        vec4,   attribute #0
// normal,          vec3,   attribute #1
// texture coord,   vec3,   attribute #2
// tangent,         vec3,   attribute #3
//
// An instance of any of these shapes is create with a single call:
//    unsigned int obj = CreateSphere(divisions, &quadCount);
// and drawn by:
//    glBindVertexArray(vaoID);
//    glDrawElements(GL_TRIANGLES, vertexcount, GL_UNSIGNED_INT, 0);
//    glBindVertexArray(0);
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <fstream>
#include <stdlib.h>

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;

#include <glu.h>                // For gluErrorString
#define CHECKERROR {GLenum err = glGetError(); if (err != GL_NO_ERROR) { fprintf(stderr, "OpenGL error (at line shapes.cpp:%d): %s\n", __LINE__, gluErrorString(err)); exit(-1);} }

#define GLM_FORCE_RADIANS
#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "math.h"
#include "shapes.h"
#include "simplexnoise.h"

const float PI = 3.14159f;
const float rad = PI/180.0f;

void pushquad(std::vector<glm::ivec3> &Tri, int i, int j, int k, int l)
{
    Tri.push_back( glm::ivec3(i, j, k) );
    Tri.push_back( glm::ivec3(i, k, l) );
}

// Batch up all the data defining a shape to be drawn (example: the
// teapot) as a Vertex Array object (VAO) and send it to the graphics
// card.  Return an OpenGL identifier for the created VAO.
unsigned int VaoFromTris(std::vector<glm::vec4> const& Pnt,
                         std::vector<glm::vec3> const& Nrm,
                         std::vector<glm::vec2> const& Tex,
                         std::vector<glm::vec3> const& Tan,
                         std::vector<glm::ivec3> const& Tri)
{
    printf("VaoFromTris %ld %ld\n", Pnt.size(), Tri.size());
    unsigned int vaoID;
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    GLuint Pbuff;
    glGenBuffers(1, &Pbuff);
    glBindBuffer(GL_ARRAY_BUFFER, Pbuff);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * Pnt.size(), &Pnt[0][0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (Nrm.size() > 0) 
    {
        GLuint Nbuff;
        glGenBuffers(1, &Nbuff);
        glBindBuffer(GL_ARRAY_BUFFER, Nbuff);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*Nrm.size(), &Nrm[0][0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
    }

    if (Tex.size() > 0) 
    {
        GLuint Tbuff;
        glGenBuffers(1, &Tbuff);
        glBindBuffer(GL_ARRAY_BUFFER, Tbuff);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*Tex.size(), &Tex[0][0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
    }

    if (Tan.size() > 0) 
    {
        GLuint Dbuff;
        glGenBuffers(1, &Dbuff);
        glBindBuffer(GL_ARRAY_BUFFER, Dbuff);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*Tan.size(), &Tan[0][0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
    }

    GLuint Ibuff;
    glGenBuffers(1, &Ibuff);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ibuff);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * Tri.size(), &Tri[0][0], GL_STATIC_DRAW);

    glBindVertexArray(0);

    return vaoID;
}

void Shape::MakeVAO()
{
    vaoID = VaoFromTris(Pnt, Nrm, Tex, Tan, Tri);
    count = Tri.size();
}

void Shape::DrawVAO(int objectId)
{
    CHECKERROR;
    glBindVertexArray(vaoID);
    if(objectId != 5)
        glDrawElements(GL_TRIANGLES, 3 * count, GL_UNSIGNED_INT, 0);
    else
        glDrawElements(GL_LINES, 3 * count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    CHECKERROR;
}

////////////////////////////////////////////////////////////////////////
// Generates a sphere of radius 1.0 centered at the origin.
//   n specifies the number of polygonal subdivisions
Sphere::Sphere(const int n)
{
    diffuseColor = glm::vec3(0.5, 0.5, 1.0);
    specularColor = glm::vec3(1.0, 1.0, 1.0);
    shininess = 120.0;

    float d = 2.0f*PI/float(n*2);
    for (int i=0;  i<=n*2;  i++) 
    {
        float s = i*2.0f*PI/float(n*2);
        for (int j=0;  j<=n;  j++) 
        {
            float t = j*PI/float(n);
            float x = cos(s)*sin(t);
            float y = sin(s)*sin(t);
            float z = cos(t);
            Pnt.push_back(glm::vec4(x, y, z, 1.0f));
            Nrm.push_back(glm::vec3(x, y, z));
            Tex.push_back(glm::vec2(s / ( 2 * PI ), t / PI));
            Tan.push_back(glm::vec3(-sin(s), cos(s), 0.0));
            if (i>0 && j>0) 
            {
                pushquad(Tri, (i-1)*(n+1) + (j-1),
                                      (i-1)*(n+1) + (j),
                                      (i  )*(n+1) + (j),
                                      (i  )*(n+1) + (j-1)); 
            } 
        } 
    }
    MakeVAO();
}

////////////////////////////////////////////////////////////////////////
// Generates a plane with normals, texture coords, and tangent vectors
// from an n by n grid of small quads.  A single quad might have been
// sufficient, but that works poorly with the reflection map.
Plane::Plane(const float r, const int n)
{
    diffuseColor = glm::vec3(0.3, 0.2, 0.1);
    specularColor = glm::vec3(1.0, 1.0, 1.0);
    shininess = 120.0;

    for (int i=0;  i<=n;  i++) 
    {
        float s = i/float(n);
        for (int j=0;  j<=n;  j++) 
        {
            float t = j/float(n);
            Pnt.push_back(glm::vec4(s*2.0*r-r, t*2.0*r-r, 0.0, 1.0));
            Nrm.push_back(glm::vec3(0.0, 0.0, 1.0));
            Tex.push_back(glm::vec2(s, t));
            Tan.push_back(glm::vec3(1.0, 0.0, 0.0));
            if (i>0 && j>0) 
            {
                pushquad(Tri, (i-1)*(n+1) + (j-1),
                                      (i-1)*(n+1) + (j),
                                      (i  )*(n+1) + (j),
                                      (i  )*(n+1) + (j-1)); 
            } 
        } 
    }
    MakeVAO();
}

////////////////////////////////////////////////////////////////////////
// Generates a plane with normals, texture coords, and tangent vectors
// from an n by n grid of small quads.  A single quad might have been
// sufficient, but that works poorly with the reflection map.
ProceduralGround::ProceduralGround(const float _range, const int n,
                     const float _octaves, const float _persistence, const float _scale,
                     const float _low, const float _high)
    :range(_range), octaves(_octaves), persistence(_persistence), scale(_scale), 
     low(_low), high(_high)
{
    diffuseColor = glm::vec3(0.3, 0.2, 0.1);
    specularColor = glm::vec3(1.0, 1.0, 1.0);
    shininess = 10.0;
    specularColor = glm::vec3(0.0, 0.0, 0.0);
    xoff = range*( time(NULL)%1000 );

    float h = 0.001;
    for (int i=0;  i<=n;  i++) 
    {
        float s = i/float(n);
        for (int j=0;  j<=n;  j++) 
        {
            float t = j/float(n);
            float x = s*2.0*range-range;
            float y = t*2.0*range-range;
            float z = HeightAt(x, y);
            float zu = HeightAt(x+h, y);
            float zv = HeightAt(x, y+h);
            Pnt.push_back(glm::vec4(x, y, z, 1.0));
            glm::vec3 du(1.0, 0.0, (zu-z)/h);
            glm::vec3 dv(0.0, 1.0, (zv-z)/h);
            Nrm.push_back(glm::normalize(glm::cross(du,dv)));
            Tex.push_back(glm::vec2(s, t));
            Tan.push_back(glm::vec3(1.0, 0.0, 0.0));
            if (i>0 && j>0) 
            {
                pushquad(Tri,
                         (i-1)*(n+1) + (j-1),
                         (i-1)*(n+1) + (j),
                         (i  )*(n+1) + (j),
                         (i  )*(n+1) + (j-1)); 
            } 
        } 
    }
    MakeVAO();
}

float ProceduralGround::HeightAt(const float x, const float y)
{
    glm::vec3 highPoint = glm::vec3(0.0, 0.0, 0.01);

    float rs = glm::smoothstep(range-20.0f, range, sqrtf(x*x+y*y));
    float noise = scaled_octave_noise_2d(octaves, persistence, scale, low, high, x+xoff, y);
    float z = (1-rs)*noise + rs*low;
    
    float hs = glm::smoothstep(15.0f, 45.0f,
                               glm::l2Norm(glm::vec3(x,y,0)-glm::vec3(highPoint.x,highPoint.y,0)));
    return (1-hs)*highPoint.z + hs*z;
}

Cube::Cube()
{
    diffuseColor = glm::vec3(0.5, 0.5, 1.0);
    specularColor = glm::vec3(1.0, 1.0, 1.0);
    shininess = 120.0;

    // adding 8 points to Pnt vectors
    Pnt.push_back(glm::vec4(  0.5f, -0.5f,  0.5f, 1.0f )); // 0   
    Pnt.push_back(glm::vec4( -0.5f, -0.5f,  0.5f, 1.0f )); // 1

    Pnt.push_back(glm::vec4(  0.5f,  0.5f,  0.5f, 1.0f )); // 2
    Pnt.push_back(glm::vec4( -0.5f,  0.5f,  0.5f, 1.0f )); // 3

    Pnt.push_back(glm::vec4(  0.5f, -0.5f, -0.5f, 1.0f )); // 4
    Pnt.push_back(glm::vec4( -0.5f, -0.5f, -0.5f, 1.0f )); // 5

    Pnt.push_back(glm::vec4(  0.5f,  0.5f, -0.5f, 1.0f )); // 6
    Pnt.push_back(glm::vec4( -0.5f,  0.5f, -0.5f, 1.0f )); // 7

    // Front face
    Tri.push_back(glm::ivec3( 0, 2, 1 ));
    Tri.push_back(glm::ivec3( 1, 2, 3 ));

    // Back face
    Tri.push_back(glm::ivec3( 5, 7, 6 ));
    Tri.push_back(glm::ivec3( 5, 6, 4 ));

    // Right face
    Tri.push_back(glm::ivec3( 4, 6, 0 ));
    Tri.push_back(glm::ivec3( 0, 6, 2 ));

    // Left face
    Tri.push_back(glm::ivec3( 1, 3, 5 ));
    Tri.push_back(glm::ivec3( 5, 3, 7 ));

    // Top face
    Tri.push_back(glm::ivec3( 6, 7, 2 ));
    Tri.push_back(glm::ivec3( 2, 7, 3 ));

    // Btm face
    Tri.push_back(glm::ivec3( 5, 4, 1 ));
    Tri.push_back(glm::ivec3( 1, 4, 0 ));

    MakeVAO();
}
