/*!
@file       YourRasterizer.cpp
@author     Prasanna Ghali       (pghali@digipen.edu)
@co-author  Wong Man Cong        (w.mancong@digipen.edu)

CVS: $Id: YourRasterizer.cpp,v 1.13 2005/03/15 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourRasterizer.h"

/*                                                                  functions
----------------------------------------------------------------------------- */
/*  _________________________________________________________________________ */
/*! Render a triangle in filled mode using 3 device-frame points.
  @param dev <->  Pointer to the pipe to render to.
  @param v0  -->  First vertex.
  @param v1  -->  Second vertex.
  @param v2  -->  Third vertex.

READ THIS FUNCTION HEADER CAREFULLY:

Begin by reading the declaration of type gfxVertex (in Vertex.h) so that
you're aware of the data encapsulated by a gfxVertex object.

-------------------------------------------------------------------------------
Triangles can be rendered in one of three ways:
1) Texture mapped (but not lit)
2) Lit (but not texture mapped)
3) Lit and texture mapped

To reduce computational overheads, this software pipe doesn't implement both
lighting and texture mapping on objects - it is one or the other.
The current rendering state of the triangle (or object) is determined by
the following call:

  unsigned int  *tex = dev->GetCurrentTexture();

If the function returns 0 (a null pointer), there is no texture state bound
to the currently-drawing triangle. This means that the object is lit but not
texture mapped. Otherwise if the function returns an address, the object
is texture mapped but not lit. The texture image is buffered in memory as
a sequence of 32-bit unsigned integral values in xxRRGGBB format which
can be directly written to the colorbuffer without any transformations.
The width and height of the texture image can be determined by the calls:

 unsigned int  tW = dev->GetCurrentTextureWidth();
 unsigned int  tH = dev->GetCurrentTextureHeight();

NOTE: Assume that GL_REPEAT is the default wrapping mode for both s and t
texture coordinates.
-------------------------------------------------------------------------------

Testing depth buffer state of the pipe:
Users can choose to turn on or off the depth buffering algorithm. The current
state of the pipe can be determined by the following call:

  bool          wdepth    = dev->DepthTestEnabled();

If DepthTestEnabled() returns 0, depth buffering is disabled, otherwise
it is enabled.
-------------------------------------------------------------------------------

Which interpolation scheme is to be used by the rasterizer to interpolate
vertex color and texture coordinates?
Since users can set the graphics pipe's to render scenes using linear or
hyperbolic interpolation, your rasterizer must implement both these
interpolation schemes. The current interpolation state of the graphics pipe
can be determined by the call:

  gfxInterpMode im = dev->GetInterpMode();

Enumeration type gfxInterpMode is declared in GraphicsPipe.h with
enumeration constants: GFX_LINEAR_INTERP or GFX_HYPERBOLIC_INTERP.
-------------------------------------------------------------------------------

How is the front colorbuffer accessed?
The dimensions of the colorbuffer are specified by the user and have values
determined by the read-only sWindowWidth and sWindowHeight variables defined
at file-scope in main.cpp. The calls

  unsigned int  *frameBuf = dev->GetFrameBuffer();
  size_t        w         = dev->GetWidth();
  size_t        h         = dev->GetHeight();

returns the address of the first element, the width, and the height of the
of the colorbuffer. The colorbuffer is a linear array of 32-bit unsigned
integral values. The pixel form is xxRRGGBB - the most significant 8 bits
are unused. To write to this colorbuffer, you must convert the normalized
color components in range [0.f, 1.f] from the interpolator into values in
range [0, 255] and pack the components into a 32-bit integral value with
format xxRRGGBB.
-------------------------------------------------------------------------------

How is the depthbuffer accessed?
The depthbuffer's dimensions are exactly the same as the colorbuffer. The call

  float         *depthBuf = dev->GetDepthBuffer();

returns the address of the first element of the depthbuffer. As can be seen the
depthbuffer stores depthvalues in the range [0.f, 1.f]. The depthbuffer
has the same dimensions as the colorbuffer.
-------------------------------------------------------------------------------

What are the coordinate conventions of the colorbuffer, depthbuffer and
texels in the texture image?
Memory for these buffer is allocated by Windows which is also used to blit
the contents of the colorbuffer to the display device's video memory.
Windows defines the colorbuffer with the convention that the origin is at
the upper-left corner. This means that assigning the value 0x00ff0000 to
frameBuf[0] will paint the upper-left corner with red color. Reading a
depth value from the first location of the depthbuffer provides the depth
value of the upper-left corner. Simiarly, reading a texel from the first
location of the texture image will return the texel associated with the
upper-left corner. However, the graphics pipe simulates OpenGL and therefore
generates device coordinates with the bottom-left corner as the origin. All of
this means that it is your responsibility to map the OpenGL device (or viewport
or window) coordinates from the bottom-left corner to upper-left corner.
*/
void YourRasterizer::DrawFilled(gfxGraphicsPipe *dev, gfxVertex const&	v0, gfxVertex const& v1, gfxVertex const&	v2)
{
    glm::vec3 const pos[3] =
    {
        { v0.x_d, v0.y_d, v0.z_d },
        { v1.x_d, v1.y_d, v1.z_d },
        { v2.x_d, v2.y_d, v2.z_d }
    };

    float A = triangle_area(pos) * 0.5f;
    if (backface_cull(A))
        return;

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
    min.x = static_cast<int>( std::floor(std::min(pos[0].x, std::min(pos[1].x, pos[2].x) ) ) );
    min.y = static_cast<int>( std::floor(std::min(pos[0].y, std::min(pos[1].y, pos[2].y) ) ) );
    max.x = static_cast<int>( std::ceil (std::max(pos[0].x, std::max(pos[1].x, pos[2].x) ) ) );
    max.y = static_cast<int>( std::ceil (std::max(pos[0].y, std::max(pos[1].y, pos[2].y) ) ) );

    // evaulation of pixel to check if it's part of the triangle
    auto evaluation_value = [](Edge const& e, glm::vec2 const& p)
    {
        return e.a * p.x + e.b * p.y + e.c;
    };

    glm::vec2 const pos0 = glm::vec2(static_cast<float>(min.x) + 0.5f, static_cast<float>(min.y) + 0.5f);
    float vEval0 = evaluation_value(tri.e0, pos0);
    float vEval1 = evaluation_value(tri.e1, pos0);
    float vEval2 = evaluation_value(tri.e2, pos0);

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

    // check to see to perform for texture mapping or not
    // 0: smooth rendering, 1: textured rendering
    size_t const TEXTURED = dev->GetCurrentTexture() ? 1 : 0;

    // Calculate the new area to use for the calculation of color when linearly interpolating thru the vertices
    A = 1.0f / (2.0f * A);
    // Vertical area of triangle
    float vTri0 = vEval0 * A, vTri1 = vEval1 * A, vTri2 = vEval2 * A;
    float const tri0_x_inc = tri.e0.a * A, tri1_x_inc = tri.e1.a * A, tri2_x_inc = tri.e2.a * A;
    float const tri0_y_inc = tri.e0.b * A, tri1_y_inc = tri.e1.b * A, tri2_y_inc = tri.e2.b * A;

    auto TexelColor = [&](float hTri0, float hTri1, float hTri2)
    {
        // texture coordinates
        glm::vec2 const& t0 = { v0.s, v0.t },
                         t1 = { v1.s, v1.t },
                         t2 = { v2.s, v2.t };

        // interpolate texture coordinate
        glm::vec2 const& tex_coord = hTri0 * t0 + hTri1 * t1 + hTri2 * t2;
        return GetTexel(dev, tex_coord);
    };

    auto set_pixel = [&](int x, int y, float z, Color clr)
    {
        if (dev->DepthTestEnabled())
            set_pixel_color(dev, x, y, z, clr);
        else
            set_pixel_color(dev, x, y, clr);
    };

    Color clr{ 0 };
    for (int y = min.y; y < max.y; ++y)
    {
        float hEval0 = vEval0, hEval1 = vEval1, hEval2 = vEval2;
        float hTri0 = vTri0, hTri1 = vTri1, hTri2 = vTri2;
        for (int x = min.x; x < max.x; ++x)
        {
            if (point_in_triangle(hEval0, hEval1, hEval2))
            {
                // interpolates the depth
                float const z = (hTri0 * v0.z_d + hTri1 * v1.z_d + hTri2 * v2.z_d);

                switch (TEXTURED)
                {
                    // Not textured
                    case 0:
                    {
                        glm::vec3 const clr0{ v0.r, v0.g, v0.b }, clr1{ v1.r, v1.g, v1.b }, clr2{ v2.r, v2.g, v2.b };
                        // color value 
                        glm::vec3 const cv{ 255.0f * (clr0 * hTri0 + clr1 * hTri1 + clr2 * hTri2) };
                        clr = create_color(cv);
                        set_pixel(x, y, z, clr);
                        break;
                    }
                    // Textured
                    case 1:
                    {
                        clr = TexelColor(hTri0, hTri1, hTri2);
                        set_pixel(x, y, z, clr);
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


/*  _________________________________________________________________________ */
/*! Render a device frame point.

    @param dev -->  Pointer to the pipe to render to.
    @param v0  -->  Vertex with device frame coordinates previously computed.

    The framebuffer is a linear array of 32-bit pixels.
    fb[y * width + x] will access a pixel at (x, y). The
    pixel format is xxRRGGBB (thus the shifting that's
    going on to pack the color components into a single
    32-bit pixel value.
*/
void YourRasterizer::DrawPoint(gfxGraphicsPipe *dev, gfxVertex const&	v0)
{
    Pos x = static_cast<Pos>(v0.x_d), y = static_cast<Pos>(v0.y_d);
    set_pixel_color(dev, x, y);
}

/*  _________________________________________________________________________ */
/*! Render a line segment between 2 device frame points.

    @param dev -->  Pointer to the pipe to render to.
    @param v0  -->  First vertex.
    @param v1  -->  Second vertex.
*/
void YourRasterizer::DrawLine(gfxGraphicsPipe *dev, gfxVertex const&	v0, gfxVertex const&	v1)
{
    int x1 = static_cast<int>(v0.x_d),
        y1 = static_cast<int>(v0.y_d),
        x2 = static_cast<int>(v1.x_d),
        y2 = static_cast<int>(v1.y_d);
    int dx = x2 - x1, dy = y2 - y1;

    if (!dx && !dy)
        return;

    int xstep = dx < 0 ? -1 : 1, ystep = dy < 0 ? -1 : 1;

    dx = dx < 0 ? -dx : dx;
    dy = dy < 0 ? -dy : dy;

    set_pixel_color(dev, x1, y1);

    if (dx >= dy)
    {
        int dk = (dy << 1) - dx, dmin = dy << 1, dmaj = (dy << 1) - (dx << 1);
        while (--dx)
        {
            y1 += dk > 0 ? ystep : 0;
            dk += dk > 0 ? dmaj : dmin;
            x1 += xstep;
            set_pixel_color(dev, x1, y1);
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
            set_pixel_color(dev, x1, y1);
        }
    }
    return;
}

/*  _________________________________________________________________________ */
/*! Render a triangle in wireframe mode using 3 device-frame points.

    @param dev -->  Pointer to the pipe to render to.
    @param v0  -->  First vertex.
    @param v1  -->  Second vertex.
    @param v2  -->  Third vertex.
*/
void YourRasterizer::DrawWireframe(gfxGraphicsPipe* dev,
              gfxVertex const& v0,
              gfxVertex const& v1,
              gfxVertex const& v2)
{
    glm::vec3 pos[3] =
    {
        { v0.x_d, v0.y_d, v0.z_d },
        { v1.x_d, v1.y_d, v1.z_d },
        { v2.x_d, v2.y_d, v2.z_d }
    };

    if (backface_cull(pos))
        return;

    DrawLine(dev, v0, v1);
    DrawLine(dev, v1, v2);
    DrawLine(dev, v2, v0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////            MY OWN FUNCTIONS            //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/*  _________________________________________________________________________ */
/*! set_pixel_color

@param	dev: Pointer to the pipe to render to. 
        x: x coordinate
        y: y coordinate
        clr: clr at (x,y) coordinate

@return none

set a particular color at (x,y) coordinate
*/
void YourRasterizer::set_pixel_color(gfxGraphicsPipe* dev, Pos x, Pos y, Color clr)
{
    unsigned int* fb = dev->GetFrameBuffer();
    size_t const w = dev->GetWidth(), h = dev->GetHeight();

    if (0 > x || w <= x || 0 > y || h <= y)
        return;
    size_t const index = x + (h - y - 1) * w;
    fb[index] = clr;
}

/*  _________________________________________________________________________ */
/*! set_pixel_color

@param	dev: Pointer to the pipe to render to. 
        x: x coordinate
        y: y coordinate
        z: z coordinate
        clr: clr at (x,y) coordinate

@return none

set a particular color at (x,y) coordinate, checks with depth buffer before setting the color
at (x,y) coordinate
*/
void YourRasterizer::set_pixel_color(gfxGraphicsPipe* dev, Pos x, Pos y, float z, Color clr)
{
    float* dp = dev->GetDepthBuffer();
    size_t const w = dev->GetWidth(), h = dev->GetHeight();

    if (0 > x || w <= x || 0 > y || h <= y)
        return;
    size_t const index = x + y * w;
    if (*(dp + index) > z)
    {
        *(dp + index) = z;
        set_pixel_color(dev, x, y, clr);
    }
}

/*  _________________________________________________________________________ */
/*!	create_color

@param	val: color value to be used to buffer a 32-bit unsigned int in the form
             0xAARRGGBB (Alpha values will not be used)

@return a 32-bit value containing the color value in the form 0xAARRGGBB
*/
YourRasterizer::Color YourRasterizer::create_color(glm::vec3 val)
{
    Color clr{ 0 };
    float const arr[3] = { val.x, val.y, val.z };
    for (size_t i = 0, bit_wise = 16; i < 3; ++i, bit_wise -= 8)
        clr |= static_cast<Color>(*(arr + i)) << bit_wise;
    return clr;
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
float YourRasterizer::triangle_area(glm::vec3 const pos[3])
{
    return (pos[1].x - pos[0].x) * (pos[2].y - pos[0].y) - (pos[2].x - pos[0].x) * (pos[1].y - pos[0].y);
}

/*  _________________________________________________________________________ */
/*! backface_cull

@param	area: area of the triangle

@return true if area is positive, false if negative

checks if the area is positive or negative
*/
bool YourRasterizer::backface_cull(float area)
{
    return area < 0.0f;
}

/*  _________________________________________________________________________ */
/*! backface_cull

@param	pos: an array of 3 vec2 storing the window coordinate of the triangle

@return true if area is positive, false if negative

checks if the area is positive or negative
*/
bool YourRasterizer::backface_cull(glm::vec3 const pos[3])
{
    return backface_cull(triangle_area(pos));
}

/*  _________________________________________________________________________ */
/*! GetTexel

@param	dev: Pointer to the pipe to render to. 
        coord: interpolated texture coordinate 

@return Color value at coord based on the texture

Get the texel color based off the interpolated texture coordinates
*/
YourRasterizer::Color YourRasterizer::GetTexel(gfxGraphicsPipe* dev, glm::vec2 const& coord)
{
    // Texture render mode - Repeat
    glm::vec2 const inter_coords{ coord.x - static_cast<int>(coord.x), coord.y - static_cast<int>(coord.y) };
    // Getting the texture width and height
    Pos const tw = dev->GetCurrentTextureWidth(), th = dev->GetCurrentTextureHeight();
    size_t const s = static_cast<size_t>( std::floor(inter_coords.s * tw) );
    size_t const t = static_cast<size_t>( std::floor(inter_coords.t * th) );

    // return white if s or t more than texture's width and height
    if (static_cast<size_t>(tw) <= s || static_cast<size_t>(th) <= t)
        return 0xFFFFFFFF;
    size_t const index = s + t * tw;
    return *(dev->GetCurrentTexture() + index);
}
