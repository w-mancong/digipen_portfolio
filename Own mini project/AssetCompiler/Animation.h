#pragma once

#include <glm/glm.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <string>
#include <vector>
#include <map>

#include "SkinnedModel.h"

namespace Animation
{
	struct AssimpNodeData
	{
		glm::mat4 transformation{};
		std::string name{};
		int childrenCount{};
		std::vector<AssimpNodeData> children{};
	};
}