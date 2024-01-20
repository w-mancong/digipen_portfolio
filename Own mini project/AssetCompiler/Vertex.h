#pragma once
#include <glm/glm.hpp>

struct Vertex
{
	glm::vec3 position{};
	glm::vec3 normal{};
	glm::vec2 uv{};
	glm::vec3 tangent{};
	glm::vec3 bitangent{};
	glm::ivec4 boneIDs{ -1, -1, -1, -1 };
	glm::vec4 weights{ 0.0f, 0.0f, 0.0f, 0.0f };

	bool operator==(const Vertex& v) const&
	{
		return position == v.position
			&& normal == v.normal
			&& uv == v.uv
			&& tangent == v.tangent
			&& bitangent == v.bitangent
			&& boneIDs == v.boneIDs
			&& weights == v.weights;
	}
};