/*!
@file       YourRasterizer.h
@author     Prasanna Ghali       (pghali@digipen.edu)
@co-author  Wong Man Cong        (w.mancong@digipen.edu)

CVS: $Id: YourRasterizer.h,v 1.13 2005/03/15 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#ifndef YOUR_RASTERIZER_H_
#define YOUR_RASTERIZER_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "gfx/GFX.h"
#include <glm/glm.hpp>

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class YourRasterizer : public gfxController_Rasterization
/*! Example rasterizer subclass.
    
    An instance of this subclass can be created and used to override the
    reference rasterizer in a specific pipe by calling the UseRasterizer()
    method of gfxGraphicsPipe. 
    
    You can set the render mode of a pipe by calling SetRenderMode() and
    passing either GFX_POINTS, GFX_WIREFRAME, or GFX_FILLED. The appropriate
    function from this rasterizer subclass will be invoked for each triangle.
    
    Within each function, dev points to the pipe to which the triangle should
    be rasterized. You can get a pointer to the framebuffer and its width and
    height from there. Each vertex contains x_d and y_d fields (X and Y screen
    position) as well as an z_d field (depth value), r, g, b color fields, and
		u, v texture coordinate fields.
    
    The pipe framebuffer is an array of unsigned ints in ARGB format. The
    examples below should be enough to get you started.
*/
{
  public:
    // ct and dt
             YourRasterizer() { }
    virtual ~YourRasterizer() { }

    // operations
    virtual void DrawPoint(gfxGraphicsPipe*, gfxVertex const&);
	virtual void DrawLine(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&);
	virtual void DrawWireframe(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&, gfxVertex const&);
	virtual void DrawFilled(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&, gfxVertex const&);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////            MY OWN FUNCTIONS            //////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    struct Edge
    {
        float a{ 0.0f }, b{ 0.0 }, c{ 0.0 };
        bool tl{ false };
    };

    struct Triangle
    {
        Edge e0, e1, e2;
    };

    using Color = unsigned int; using Pos = unsigned int;
    /*  _________________________________________________________________________ */
    /*! set_pixel_color

    @param	dev: Pointer to the pipe to render to.
            x: x coordinate
            y: y coordinate
            clr: clr at (x,y) coordinate

    @return none

    set a particular color at (x,y) coordinate
    */
    void set_pixel_color(gfxGraphicsPipe* dev, Pos x, Pos y, Color clr = 0);

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
    void set_pixel_color(gfxGraphicsPipe* dev, Pos x, Pos y, float z, Color clr = 0);

    /*  _________________________________________________________________________ */
    /*!	create_color

    @param	val: color value to be used to buffer a 32-bit unsigned int in the form
                 0xAARRGGBB (Alpha values will not be used)

    @return a 32-bit value containing the color value in the form 0xAARRGGBB
    */
    Color create_color(glm::vec3 val);

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
    float triangle_area(glm::vec3 const pos[3]);

    /*  _________________________________________________________________________ */
    /*! backface_cull

    @param	area: area of the triangle

    @return true if area is positive, false if negative

    checks if the area is positive or negative
    */
    bool backface_cull(float area);

    /*  _________________________________________________________________________ */
    /*! backface_cull

    @param	pos: an array of 3 vec2 storing the window coordinate of the triangle

    @return true if area is positive, false if negative

    checks if the area is positive or negative
    */
    bool backface_cull(glm::vec3 const pos[3]);

    /*  _________________________________________________________________________ */
    /*! GetTexel

    @param	dev: Pointer to the pipe to render to.
            coord: interpolated texture coordinate

    @return Color value at coord based on the texture

    Get the texel color based off the interpolated texture coordinates
    */
    Color GetTexel(gfxGraphicsPipe* dev, glm::vec2 const& coord);
};

#endif  /* YOUR_RASTERIZER_H_ */
