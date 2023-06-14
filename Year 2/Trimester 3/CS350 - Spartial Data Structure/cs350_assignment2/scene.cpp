////////////////////////////////////////////////////////////////////////
// The scene class contains all the parameters needed to define and
// draw a simple scene, including:
//   * Geometry
//   * Light parameters
//   * Material properties
//   * viewport size parameters
//   * Viewing transformation values
//   * others ...
//
// Some of these parameters are set when the scene is built, and
// others are set by the framework in response to user mouse/keyboard
// interactions.  All of them can be used to draw the scene.

#include "math.h"
#include <iostream>
#include <stdlib.h>

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;

#include <glu.h>                // For gluErrorString

#define GLM_FORCE_RADIANS
#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/ext.hpp>          // For printing GLM objects with to_string

#include "framework.h"
#include "shapes.h"
#include "object.h"
#include "texture.h"
#include "transform.h"
#include <random>

#if 0
std::random_device randomSeed;
std::mt19937_64 RG(randomSeed());
#else
std::string seed = "A constant seed.";
std::seed_seq constSeed (seed.begin(),seed.end());
std::mt19937_64 RG(constSeed);
#endif

std::uniform_real_distribution<> myrandom(0.0, 1.0);
std::uniform_real_distribution<> BALrandom(-1.0, 1.0);
// Call myrandom(RG) to get a uniformly distributed random number in [0,1].

const float rad = PI/180.0f;    // Convert degrees to radians

glm::mat4 Identity;

const float grndSize = 100.0;    // Island radius;  Minimum about 20;  Maximum 1000 or so
const float grndOctaves = 4.0;  // Number of levels of detail to compute
const float grndFreq = 0.03;    // Number of hills per (approx) 50m
const float grndPersistence = 0.03; // Terrain roughness: Slight:0.01  rough:0.05
const float grndLow = -3.0;         // Lowest extent below sea level
const float grndHigh = 5.0;        // Highest extent above sea level

////////////////////////////////////////////////////////////////////////
// This macro makes it easy to sprinkle checks for OpenGL errors
// throughout your code.  Most OpenGL calls can record errors, and a
// careful programmer will check the error status *often*, perhaps as
// often as after every OpenGL call.  At the very least, once per
// refresh will tell you if something is going wrong.
#define CHECKERROR {GLenum err = glGetError(); if (err != GL_NO_ERROR) { fprintf(stderr, "OpenGL error (at line scene.cpp:%d): %s\n", __LINE__, gluErrorString(err)); exit(-1);} }

// Create an RGB color from human friendly parameters: hue, saturation, value
glm::vec3 HSV2RGB(const float h, const float s, const float v)
{
    if (s == 0.0)
        return glm::vec3(v,v,v);

    int i = (int)(h*6.0) % 6;
    float f = (h*6.0f) - i;
    float p = v*(1.0f - s);
    float q = v*(1.0f - s*f);
    float t = v*(1.0f - s*(1.0f-f));
    if      (i == 0)  return glm::vec3(v,t,p);
    else if (i == 1)  return glm::vec3(q,v,p);
    else if (i == 2)  return glm::vec3(p,v,t);
    else if (i == 3)  return glm::vec3(p,q,v);
    else if (i == 4)  return glm::vec3(t,p,v);
    else   /*i == 5*/ return glm::vec3(v,p,q);
}


////////////////////////////////////////////////////////////////////////
// InitializeScene is called once during setup to create all the
// textures, shape VAOs, and shader programs as well as setting a
// number of other parameters.
void Scene::InitializeScene()
{
    glEnable(GL_DEPTH_TEST);
    CHECKERROR;

    // Set initial light parameters
    lightSpin = 150.0;
    lightTilt = 5.0;
    lightDist = 100.0;
    
    w_down = false;
    s_down = false;
    a_down = false;
    d_down = false;
    nav = true;
    spin = 0.0;
    tilt = 0.0;
    eye = glm::vec3(0.0, -5.0, 1.5);
    speed = 300.0/30.0;
    last_time = glfwGetTime();
    tr = glm::vec3(0.0, 0.0, 25.0);
   
    show_demo_window = false;   // Set to see ImGUI's demo window of all GUI elements.

    ry = 0.4;
    front = 0.5;
    back = 5000.0;

    CHECKERROR;
    objectRoot = new Object(NULL, nullId);

    // @@ These control the number of ellipsoids and triangles per ellipsoid:
    // On my laptop, with naive collision detection:
    //    12 and 6 ( 97344 triangles) runs at      144 FPS
    //    24 and 6 (389376 triangles) slows down to 92 FPS.
    //    36 and 6 (876096 triangles) slows down to 44 FPS.
    polyCount = 12;             // Produces 4*polyCount^2 triangles per ellipsoid
    ellipsoidCount = 6;         // Produces (2*ellipsoidCount+1)^2 ellipsoids

    // Create the lighting shader program from source code files.
    lightingProgram = new ShaderProgram();
    lightingProgram->AddShader("lightingPhong.vert", GL_VERTEX_SHADER);
    lightingProgram->AddShader("lightingPhong.frag", GL_FRAGMENT_SHADER);

    glBindAttribLocation(lightingProgram->programId, 0, "vertex");
    glBindAttribLocation(lightingProgram->programId, 1, "vertexNormal");
    glBindAttribLocation(lightingProgram->programId, 2, "vertexTexture");
    glBindAttribLocation(lightingProgram->programId, 3, "vertexTangent");
    lightingProgram->LinkProgram();
    
    // Create all the Polygon shapes
    proceduralground = new ProceduralGround(grndSize, 400,
                                     grndOctaves, grndFreq, grndPersistence,
                                     grndLow, grndHigh);
    
    Shape* SpherePolygons = new Sphere(polyCount);
    Shape* SeaPolygons = new Plane(2000.0, 64);
    Shape* GroundPolygons = proceduralground;

    // Various colors used in the subsequent models
    glm::vec3 woodColor(87.0/255.0, 51.0/255.0, 35.0/255.0);
    glm::vec3 brickColor(134.0/255.0, 60.0/255.0, 56.0/255.0);
    glm::vec3 floorColor(6*16/255.0, 5.5*16/255.0, 3*16/255.0);
    glm::vec3 brassColor(0.5, 0.5, 0.1);
    glm::vec3 grassColor(62.0/255.0, 102.0/255.0, 38.0/255.0);
    glm::vec3 waterColor(0.3, 0.3, 1.0);

    glm::vec3 black(0.0, 0.0, 0.0);
    glm::vec3 brightSpec(0.5, 0.5, 0.5);
    glm::vec3 polishedSpec(0.3, 0.3, 0.3);
 
    // Creates all the models from which the scene is composed.  Each
    // is created with a polygon shape (possibly NULL), a
    // transformation, and the surface lighting parameters Kd, Ks, and
    // alpha.

    sky        = new Object(SpherePolygons, skyId, black, black, 0);
    ground     = new Object(GroundPolygons, groundId, grassColor, black, 1);
    sea        = new Object(SeaPolygons, seaId, waterColor, brightSpec, 120);
    objects    = new Object(nullptr, nullId);

    for (int i=-ellipsoidCount;  i<=ellipsoidCount;  i++) 
    {
        for (int j=-ellipsoidCount;  j<=ellipsoidCount;  j++) 
        {
            glm::vec3 hue = HSV2RGB( myrandom(RG), 1.0, 1.0 );
            float r = 0.5f *( 0.5f + 2.0f * myrandom(RG) );
            float h = 2.0f * myrandom(RG);
            glm::vec3 s(r, r, h);
            int v = (i == 0 && j == 0) ? 0 : 1;
            glm::vec3 t(5.0f * i + v * 2.0f * BALrandom(RG),
                        5.0f * j + v * 2.0f * BALrandom(RG),
                        0.0);
            t[2] = proceduralground->HeightAt(t[0], t[1]) + s[2] * 0.9;
            Object* ellipsoid = new Object( SpherePolygons, 4, hue, glm::vec3(0.5), 120 );
            objects->add( ellipsoid, Translate(t) * Scale(s) ); 
        } 
    }

    // Scene is composed of some scene objects, and the sky, ground, and sea
    objectRoot->add(objects);
    objectRoot->add(sky, Scale(2000.0, 2000.0, 2000.0));
    objectRoot->add(sea); 
    objectRoot->add(ground);

    CHECKERROR;

    // @@ At this point, the scene is built and all the objects have
    // been sent to the graphics card as Vertex Array Objects.  This
    // is the point at which all the triangles can be retrieved and
    // eventually stored in a spatial data structure.  The sample code
    // here retrieves the triangles, but only counts them.
    triangleCount = 0;
    for (INSTANCE instance : objects->instances) 
    {
        Object* object = instance.first;
        const glm::mat4& modelTr = instance.second;
        //const glm::mat3& normalTr = glm::mat3(glm::inverse(modelTr));

        std::vector<glm::vec4>& Pnt = object->shape->Pnt;  // The objects list of vertices
        std::vector<glm::vec3>& Nrm = object->shape->Nrm;  // The object's list of normals
        std::vector<glm::ivec3>& Tri = object->shape->Tri; // The object's list of triangles

        for (glm::ivec3 tri : Tri) 
        {
            triangleCount++;
            
            //printf("%d %d %d\n", tri[0], tri[1], tri[2]);

            // @@ This triangle's indices are:
            //        tri[0], tri[1], tri[2]
            //    Its world coordinate points are:
            //        modelTr*Pnt[tri[0]], modelTr*Pnt[tri[1]], modelTr*Pnt[tri[2]]
            //    Its normals vectors are: (probably not needed)
            //        Nrm[tri[0]]*normalTr, Nrm[tri[1]]*normalTr, Nrm[tri[2]]*normalTr
        }

        // AABB
        //Object* cube = new Object(CubePolygons, debugId, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.5), 120);
        //aabb->add(cube, modelTr);
    }

    // @@ This is a good place to build the VAOs for the debug draws,
    // storing the results in the Scene instance.
    {
        std::vector<glm::vec4> pnt = { glm::vec4(0, 0, 0, 1), glm::vec4(1, 1, 1, 1) };
        std::vector<int> ind = { 0, 1 };
        segmentVao = VaoFromPoints(pnt, ind);

        for (INSTANCE& instance : objects->instances)
        {
            Object* obj = instance.first;
            glm::mat4 const& modelTr = instance.second;

            std::vector<glm::vec4>&  Pnt = obj->shape->Pnt; // The objects list of vertices
            std::vector<glm::ivec3>& Tri = obj->shape->Tri; // The object's list of triangles

            for (glm::ivec3 const& tri : Tri)
            {
                glm::vec4 const p0 = modelTr * Pnt[tri[0]], p1 = modelTr * Pnt[tri[1]], p2 = modelTr * Pnt[tri[2]];
                glm::vec3 const A = { (p0.x + p1.x + p2.x) / 3.0f, (p0.y + p1.y + p2.y) / 3.0f, (p0.z + p1.z + p2.z) / 3.0f };
                glm::vec3 const B = A + glm::cross(glm::vec3(p1 - p0), glm::vec3(p2 - p0));

                lineSegments.emplace_back( std::make_pair(A, B) );
            }
        }
    }

    {
        std::vector<glm::vec4> pnt = { glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f), glm::vec4( 1.0f, -1.0f, -1.0f, 1.0f), 
                                       glm::vec4(-1.0f,  1.0f, -1.0f, 1.0f), glm::vec4(-1.0f, -1.0f,  1.0f, 1.0f),
                                       glm::vec4( 1.0f,  1.0f, -1.0f, 1.0f), glm::vec4( 1.0f, -1.0f,  1.0f, 1.0f), 
                                       glm::vec4(-1.0f,  1.0f,  1.0f, 1.0f), glm::vec4( 1.0f,  1.0f,  1.0f, 1.0f) 
                                     };
        std::vector<int> ind       = { 0, 1, 1, 4, 4, 2, 2, 0, // front
                                       1, 5, 5, 7, 7, 4, 4, 1, // right
                                       5, 3, 3, 6, 6, 7, 7, 5, // back
                                       6, 2, 2, 0, 0, 3, 3, 6, // left
                                       4, 7, 7, 6, 6, 2, 2, 4, // top
                                       0, 1, 1, 5, 5, 3, 3, 0, // btm
                                     };

        aabbVao = VaoFromPoints(pnt, ind);
    }
}

void Scene::DrawGUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // @@ Here is where the ImGUI controls are drawn.  This example
    // displays several ImGui::Text widgets, and one ImGui::Checkbox
    // widget.
    ImGui::Text(" %.1f FPS;  %.3f ms/frame;",
                ImGui::GetIO().Framerate,
                1000.0f / ImGui::GetIO().Framerate);
    ImGui::Text(" %d triangles managed", triangleCount);

    ImGui::Checkbox("AABB", &drawAABB);
    ImGui::Checkbox("Triangle AABB", &drawTriangleAABB);
    ImGui::Checkbox("Line Segment", &drawLineSegments);

    //ImGui::Checkbox("Show_demo_window", &show_demo_window);
    //if (show_demo_window)
    //    ImGui::ShowDemoWindow();
        
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Scene::BuildTransforms()
{
    
    // Work out the eye position as the user move it with the WASD keys.
    float now = glfwGetTime();
    float dist = (now-last_time)*speed;
    last_time = now;
    glm::vec3 dir;
    if (w_down)
        dir =  glm::vec3(sin(spin*rad), cos(spin*rad), 0.0);
    if (s_down)
        dir = -glm::vec3(sin(spin*rad), cos(spin*rad), 0.0);
    if (d_down)
        dir =  glm::vec3(cos(spin*rad), -sin(spin*rad), 0.0);
    if (a_down)
        dir = -glm::vec3(cos(spin*rad), -sin(spin*rad), 0.0);

    // @@ Check if moving the eye by distance dist in direction dir is
    // legal.  Do this by traversing the spatial data structure
    // looking for collision of the segment(eye, eye+step) with any
    // object. 
    eye += dist*dir;
    eye[2] = proceduralground->HeightAt(eye[0], eye[1]) + 1.5; // Set the height of the eye

    CHECKERROR;

    if (nav)
        WorldView = Rotate(0, tilt-90)*Rotate(2, spin) *Translate(-eye[0], -eye[1], -eye[2]);
    else
        WorldView = Translate(tr[0], tr[1], -tr[2]) *Rotate(0, tilt-90)*Rotate(2, spin);
    WorldProj = Perspective((ry*width)/height, ry, front, (mode==0) ? 1000 : back);

}

////////////////////////////////////////////////////////////////////////
// Procedure DrawScene is called whenever the scene needs to be
// drawn. (Which is often: 30 to 60 times per second are the common
// goals.)
void Scene::DrawScene()
{
    // Set the viewport
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    CHECKERROR;
    // Calculate the light's position from lightSpin, lightTilt, lightDist
    lightPos = glm::vec3(lightDist*cos(lightSpin*rad)*sin(lightTilt*rad),
                         lightDist*sin(lightSpin*rad)*sin(lightTilt*rad), 
                         lightDist*cos(lightTilt*rad));

    BuildTransforms();

    // The lighting algorithm needs the inverse of the WorldView matrix
    WorldInverse = glm::inverse(WorldView);
    
    CHECKERROR;
    int loc, programId;

    ////////////////////////////////////////////////////////////////////////////////
    // Lighting pass
    ////////////////////////////////////////////////////////////////////////////////
    
    // Choose the lighting shader
    lightingProgram->UseShader();
    programId = lightingProgram->programId;

    // Set the viewport, and clear the screen
    glViewport(0, 0, width, height);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

    loc = glGetUniformLocation(programId, "WorldProj");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(WorldProj));
    loc = glGetUniformLocation(programId, "WorldView");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(WorldView));
    loc = glGetUniformLocation(programId, "WorldInverse");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(WorldInverse));
    loc = glGetUniformLocation(programId, "lightPos");
    glUniform3fv(loc, 1, &(lightPos[0]));   
    loc = glGetUniformLocation(programId, "mode");
    glUniform1i(loc, mode);
    CHECKERROR;

    // Draw all objects (This recursively traverses the object hierarchy.)
    CHECKERROR;
    // @@ To avoid z-fighting between the scene's triangles, and your
    // debug drawing of triangles, surround this draw command with
    // enable/disable GL_POLYGON_OFFSET_FILL
    
    // glPolygonOffset(1.0, polygonOffset);
    // glEnable(GL_POLYGON_OFFSET_FILL);
    objectRoot->Draw(lightingProgram, Identity);
    // glDisable(GL_POLYGON_OFFSET_FILL);
    CHECKERROR;

    // Turn off the shader
    lightingProgram->UnuseShader();

    // @@ The main image is drawn by this point.  This is a good time
    // to draw any debug drawing items.
    if(drawLineSegments)
        DrawLineSegment();

    if(drawAABB)
        DrawAABB();

    if(drawTriangleAABB)
        DrawTriangleAABB();

    ////////////////////////////////////////////////////////////////////////////////
    // End of Lighting pass
    ////////////////////////////////////////////////////////////////////////////////
}

void Scene::DestroyScene()
{
    // @@ This is called as the program is exiting. Perform any
    // necessary cleanup here.
}

unsigned int Scene::VaoFromPoints(std::vector<glm::vec4> pnt, std::vector<int> ind)
{
    unsigned int vaoID;
    glGenVertexArrays(1, &vaoID);
    glBindVertexArray(vaoID);

    GLuint pBuf;
    glGenBuffers(1, &pBuf);
    glBindBuffer(GL_ARRAY_BUFFER, pBuf);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * pnt.size(),  &pnt[0][0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint iBuf;
    glGenBuffers(1, &iBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iBuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * ind.size(), &ind[0], GL_STATIC_DRAW);
    glBindVertexArray(0);

    return vaoID;
}

void Scene::DrawLineSegment()
{
    for (std::pair<glm::vec3, glm::vec3> const& pts : lineSegments)
    {
        glm::vec3 const& A = pts.first;
        glm::vec3 const& B = pts.second;

        DrawLineSegment(A, B);
    }
}

void Scene::DrawLineSegment(glm::vec3 const& p0, glm::vec3 const& p1)
{
    glm::mat4 modelTr
    {
        { p1.x - p0.x, 0.0f, 0.0f, 0.0f },
        { 0.0f, p1.y - p0.y, 0.0f, 0.0f },
        { 0.0f, 0.0f, p1.z - p0.z, 0.0f },
        { p0.x, p0.y, p0.z, 1.0f },
    };

    int loc{}, programId{};
    lightingProgram->UseShader();
    programId = lightingProgram->programId;

    loc = glGetUniformLocation(programId, "ModelTr");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(modelTr));

    loc = glGetUniformLocation(programId, "objectId");
    glUniform1i(loc, debugId);

    glBindVertexArray(segmentVao);
    glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    lightingProgram->UnuseShader();
}

void Scene::DrawAABB()
{
    int loc{}, programId{};
    lightingProgram->UseShader();
    programId = lightingProgram->programId;

    for (INSTANCE& instances : objects->instances)
    {
        glm::mat4& modelTr = instances.second;

        loc = glGetUniformLocation(programId, "ModelTr");
        glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(modelTr));

        loc = glGetUniformLocation(programId, "objectId");
        glUniform1i(loc, debugId);

        glBindVertexArray(aabbVao);
        glDrawElements(GL_LINES, 48, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    lightingProgram->UnuseShader();
}

void Scene::DrawTriangleAABB()
{
    int loc{}, programId{};
    lightingProgram->UseShader();
    programId = lightingProgram->programId;

    for (INSTANCE& instances : objects->instances)
    {
        Object* obj = instances.first;
        glm::mat4 const& modelTr = instances.second;

        std::vector<glm::vec4>& Pnt = obj->shape->Pnt; // The objects list of vertices
        std::vector<glm::ivec3>& Tri = obj->shape->Tri; // The object's list of triangles

        for (glm::ivec3 tri : Tri)
        {
            glm::vec4 const p0 = modelTr * Pnt[tri[0]], p1 = modelTr * Pnt[tri[1]], p2 = modelTr * Pnt[tri[2]];
            glm::mat4 trans
            {
                { p1.x - p0.x, 0.0f, 0.0f, 0.0f },
                { 0.0f, p1.y - p0.y, 0.0f, 0.0f },
                { 0.0f, 0.0f, p1.z - p0.z, 0.0f },
                { p0.x, p0.y, p0.z, 1.0f },
            };

            loc = glGetUniformLocation(programId, "ModelTr");
            glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(trans));

            loc = glGetUniformLocation(programId, "objectId");
            glUniform1i(loc, debugId);

            glBindVertexArray(aabbVao);
            glDrawElements(GL_LINES, 48, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }

    lightingProgram->UnuseShader();
}
