#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "glm/glm.hpp"

namespace MeshCompiler
{
	aiTextureType constexpr const TEXTURE_TYPE[] = { aiTextureType_DIFFUSE, aiTextureType_AMBIENT, aiTextureType_EMISSIVE,
													 aiTextureType_NORMALS, aiTextureType_SHININESS, aiTextureType_METALNESS };
	uint64_t constexpr const NUM_TEXTURE_TYPE = sizeof(TEXTURE_TYPE) / sizeof(*TEXTURE_TYPE);
	char constexpr const* TEXTURE_NAMES[NUM_TEXTURE_TYPE] = { "albedo: ", "ambient_occulusion: ", "emissive: ", 
															  "normal: ", "roughness: ", "metallic: " };

	enum class Vec3Attrib
	{
		Position,
		Normals,
		Tangents,
		BiTangents,
		Total,
	};

	enum class Vec2Attrib
	{
		TextureCoords,
		Total
	};

	static constexpr uint64_t TOTAL_VEC2_ATTRIBUTE = static_cast<uint64_t>(Vec2Attrib::Total);
	static constexpr uint64_t TOTAL_VEC3_ATTRIBUTE = static_cast<uint64_t>(Vec3Attrib::Total);
	static constexpr uint64_t TOTAL_VERTEX_ATTRIBUTE = TOTAL_VEC2_ATTRIBUTE + TOTAL_VEC3_ATTRIBUTE;

	struct Submesh
	{
		std::string				 meshName{};
		std::vector<glm::vec3>   vec3Attrib[TOTAL_VEC3_ATTRIBUTE]{};
		std::vector<glm::vec2>	 vec2Attrib[TOTAL_VEC2_ATTRIBUTE]{};
		std::vector<uint32_t>	 indices{};
		std::string				 materialPaths[NUM_TEXTURE_TYPE] {};
	};

	struct DeserializationData
	{
		std::vector<Submesh> meshInfos{};
	};

	/*
		Using this struct to find the offset of each submeshes'
		vertices, indices, mesh name and material paths
	*/
	struct HeaderInfo
	{
		uint64_t meshNameSize{};
		uint64_t verticeCount{};
		uint64_t indicesCount{};
		uint64_t materialPathSize[NUM_TEXTURE_TYPE]{};
	};

	template <typename T, typename = std::is_enum<T>>
	uint64_t Idx(T att)
	{
		return static_cast<uint64_t>(att);
	}
}