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