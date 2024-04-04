/* Start Header *****************************************************************/
/*! \file (common.h)

     \author (Wong Man Cong, w.mancong, 390005621)

     \par (email: w.mancong\@digipen.edu)

     \date (March 25, 2024)

    \brief Copyright (C) 2024 DigiPen Institute of Technology.

    Reproduction or disclosure of this file or its contents without the prior written consent of DigiPen Institute of Technology is prohibited. */
/* End Header *******************************************************************/
const int HISTO_SIZE = 256;

struct HistogramUniforms
{
    float m_Cdf[HISTO_SIZE];
    uint  m_Bin[HISTO_SIZE];
};

layout(binding = 0, rgba8) uniform readonly image2D imgSample; // sample image
layout(binding = 1, rgba8) uniform image2D imgResult;          // result image
layout(std430, binding = 2) buffer HistoUniform
{
    HistogramUniforms histo;
};