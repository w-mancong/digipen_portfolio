@ECHO OFF
%VULKAN_SDK%/Bin/glslangValidator.exe -V "torus.vert" -o "torus.vert.spv"
%VULKAN_SDK%/Bin/glslangValidator.exe -V "torus.frag" -o "torus.frag.spv"
%VULKAN_SDK%/Bin/glslangValidator.exe -V "torus.tesc" -o "torus.tesc.spv"
%VULKAN_SDK%/Bin/glslangValidator.exe -V "torus.tese" -o "torus.tese.spv"
PAUSE