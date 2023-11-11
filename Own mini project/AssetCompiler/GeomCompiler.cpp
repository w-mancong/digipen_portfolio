/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				GeomCompiler.cpp
#    Primary Author:		Joachim
#    Secondary Author:		-
*********************************************************************************************************************************************************/

// --------------------
// Header Includes
// --------------------
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include "meshoptimizer.h"
#include "GeomCompiler.h"

using namespace MeshCompiler;

std::ostream& operator<<(std::ostream& os, aiString s)
{
	return os << s.C_Str();
}

std::string operator+(std::string const& lhs, aiString const& rhs)
{
	return lhs + rhs.C_Str();
}

std::string operator+(aiString const& lhs, std::string const& rhs)
{
	return lhs.C_Str() + rhs;
}

std::string operator+(aiString const& lhs, char const* rhs)
{
	return lhs.C_Str() + std::string(rhs);
}

std::string operator+(char const* lhs, aiString const& rhs)
{
	return lhs + std::string(rhs.C_Str());
}

std::ofstream& operator<<(std::ofstream& os, const aiVector3D& _vertex) {
	os << _vertex.x << "," << _vertex.y << "," << _vertex.z;
	return os;
}

std::ofstream& operator<<(std::ofstream& os, const glm::vec3& _vertex) {
	os << _vertex.x << "," << _vertex.y << "," << _vertex.z;
	return os;
}

std::ofstream& operator<<(std::ofstream& os, const glm::vec2& _vertex) {
	os << _vertex.x << "," << _vertex.y;
	return os;
}

GeomCompiler::GeomCompiler() {
	this->m_DescriptorMatrix.Scaling(aiVector3D{ 1.f, 1.f, 1.f }, this->m_DescriptorMatrix);
	this->m_DescriptorMatrix.RotationX(0.f, this->m_DescriptorMatrix);
	this->m_DescriptorMatrix.RotationY(0.f, this->m_DescriptorMatrix);
	this->m_DescriptorMatrix.RotationZ(0.f, this->m_DescriptorMatrix);
	this->m_DescriptorMatrix.Translation(aiVector3D{ 0.f, 0.f, 0.f }, this->m_DescriptorMatrix);
}

glm::vec3 ConvertaiVec3toVec3(const aiVector3D& _vector) {
	return glm::vec3{ _vector.x, _vector.y, _vector.z };
}

glm::vec2 ConvertaiVec3toVec2(const aiVector3D& _vector) {
	return glm::vec2{ _vector.x, _vector.y };
}

// ----------------------------------------
// Compile Mesh/Models
// ----------------------------------------
bool GeomCompiler::Compile(const std::string& _inputFilepath) {
	std::cout << ">> Loading mesh data from " << _inputFilepath << " ...\n";

	// Importing
	try {
		m_Scene = m_Importer.ReadFile(_inputFilepath, this->m_ImportFlag);
	}
	catch (std::exception& _e) {
		std::cout << ">> Exception countered while loading input data file\n";
		std::cout << ">> Exception: " << _e.what() << std::endl;
		return false;
	}

	if (!m_Scene) {
		std::cout << ">> Error encountered while loading mesh data\n";
		return false;
	}

	// Binary export
	auto LoadMaterial = [this](Submesh& submesh, aiMesh const* currMesh)
	{
		aiMaterial const* mat = m_Scene->mMaterials[currMesh->mMaterialIndex];
		uint64_t counter{};
		for (uint64_t typeIdx{}; typeIdx < NUM_TEXTURE_TYPE; ++typeIdx)
		{
			aiTextureType const type = *(TEXTURE_TYPE + typeIdx);
			uint32_t const matCnt = mat->GetTextureCount(type);
			for (uint32_t matIdx{}; matIdx < matCnt; ++matIdx)
			{
				aiString path{};
				aiReturn ret = mat->GetTexture(type, matIdx, &path);
				if (ret == aiReturn_FAILURE)
					continue;
				submesh.materialPaths[counter++] = (*(TEXTURE_NAMES + typeIdx) + path);
			}
		}
	};

	auto LoadVertices = [this](Submesh& submesh, aiMesh const* currMesh)
	{
		uint32_t const numVertices = currMesh->mNumVertices;
		for (uint32_t vertIdx{}; vertIdx < numVertices; ++vertIdx)
		{
			if (currMesh->HasPositions())
				submesh.vec3Attrib[Idx(Vec3Attrib::Position)].emplace_back(ConvertaiVec3toVec3(currMesh->mVertices[vertIdx]));
			if (currMesh->HasNormals())
				submesh.vec3Attrib[Idx(Vec3Attrib::Normals)].emplace_back( ConvertaiVec3toVec3( currMesh->mNormals[vertIdx] ) );
			if (currMesh->HasTangentsAndBitangents())
			{
				submesh.vec3Attrib[Idx(Vec3Attrib::Tangents)].emplace_back( ConvertaiVec3toVec3( currMesh->mTangents[vertIdx] ) );
				submesh.vec3Attrib[Idx(Vec3Attrib::BiTangents)].emplace_back( ConvertaiVec3toVec3( currMesh->mBitangents[vertIdx] ) );
			}
			if (currMesh->HasTextureCoords(0))
				submesh.vec2Attrib[Idx(Vec2Attrib::TextureCoords)].emplace_back(ConvertaiVec3toVec2(currMesh->mTextureCoords[0][vertIdx]));
		}
	};

	auto LoadIndices = [this](Submesh& submesh, aiMesh const* currMesh)
	{
		uint32_t const numFaces = currMesh->mNumFaces;
		for (uint32_t faceIdx{}; faceIdx < numFaces; ++faceIdx)
		{
			aiFace const currFace{ currMesh->mFaces[faceIdx] };
			uint32_t const indicesCnt = currFace.mNumIndices;
			for (uint32_t indicesIdx{}; indicesIdx < indicesCnt; ++indicesIdx)
				submesh.indices.emplace_back( currFace.mIndices[indicesIdx] );
		}
	};

	DeserializationData data{};
	uint32_t const numMesh = m_Scene->mNumMeshes;
	for (uint32_t cnt{}; cnt < numMesh; ++cnt)
	{
		aiMesh* currMesh{ m_Scene->mMeshes[cnt] };
		Submesh submesh{};

		// Optimize each submesh before storing the value

		submesh.meshName = currMesh->mName.C_Str();
		if (m_Scene->HasMaterials())
			LoadMaterial(submesh, currMesh);

		LoadVertices(submesh, currMesh);
		LoadIndices(submesh, currMesh);

		data.meshInfos.emplace_back(submesh);
	}

	// Exporting
	std::string outputFile{ this->m_OutputFileDirectory + _inputFilepath.substr(_inputFilepath.find_last_of('\\'), _inputFilepath.find_last_of('.') - _inputFilepath.find_last_of('\\')) + ".h_mesh" };
	return Deserialize(outputFile, data);
}

bool GeomCompiler::Deserialize(std::string const& outputFile, DeserializationData const& data)
{
	std::ofstream ofs{ outputFile, std::ios::out | std::ios::binary };
	//std::ofstream ofs{ outputFile, std::ios::out };	
	if (!ofs.is_open()) {
		std::cout << ">> Error encountered during serializiation. Output file could not be created\n";
		return false;
	}

	uint64_t const numMesh = data.meshInfos.size();
	// Write the numebr of submeshes into file
	ofs.write(reinterpret_cast<char const*>(&numMesh), sizeof(uint64_t));
	
	for (uint64_t meshIdx{}; meshIdx < numMesh; ++meshIdx)
	{
		Submesh const& submesh = data.meshInfos[meshIdx];

		HeaderInfo const info
		{
			.meshNameSize = submesh.meshName.size(),
			.verticeCount = submesh.vec3Attrib[0].size(),
			.indicesCount = submesh.indices.size()
		};

		{	// To insert the size of each material file path
			HeaderInfo& H = const_cast<HeaderInfo&>(info);
			for (uint64_t i{}; i < NUM_TEXTURE_TYPE; ++i)
				H.materialPathSize[i] = submesh.materialPaths[i].size();
		}

		// Writing the data offset for each submesh
		ofs.write(reinterpret_cast<char const*>(&info), sizeof(HeaderInfo));

		// Write the name of the mesh
		ofs.write(submesh.meshName.c_str(), info.meshNameSize);

		for (uint64_t i{}; i < TOTAL_VEC3_ATTRIBUTE; ++i)
		{	// Writing all vec3 vertex attribute into buffer
			uint64_t const SIZE = sizeof(glm::vec3) * submesh.vec3Attrib[i].size();
			ofs.write(reinterpret_cast<char const*>(submesh.vec3Attrib[i].data()), SIZE);
		}

		// Write texture coordinates
		for (uint64_t i{}; i < TOTAL_VEC2_ATTRIBUTE; ++i)
		{	// Writing all vec2 vertex attribute into buffer
			uint64_t const offset = sizeof(glm::vec2) * submesh.vec2Attrib[i].size();
			ofs.write(reinterpret_cast<char const*>(submesh.vec2Attrib[i].data()), offset);
		}

		// Write indices
		ofs.write(reinterpret_cast<char const*>( submesh.indices.data() ), sizeof(uint32_t) * info.indicesCount);

		// Write string of material path
		for (uint64_t i{}; i < NUM_TEXTURE_TYPE; ++i)
			ofs.write(submesh.materialPaths[i].c_str(), info.materialPathSize[i]);
	}

	return true;
}

/* ******************************************************************************
								Test Function
 ****************************************************************************** */
void GeomCompiler::Serialize(std::string const& inputFile)
{
	std::ifstream ifs{ inputFile, std::ios::in | std::ios::binary };
	if (!ifs.is_open()) {
		std::cout << ">> Error encountered during deserialization. Input file could not be opened\n";
		return;
	}

	uint64_t numMesh{};
	ifs.read(reinterpret_cast<char*>(&numMesh), sizeof(uint64_t));
	for (uint64_t meshIdx{}; meshIdx < numMesh; ++meshIdx)
	{
		HeaderInfo info{};
		ifs.read(reinterpret_cast<char*>(&info), sizeof(HeaderInfo));

		{	// name
			std::unique_ptr<char[]> name = std::make_unique<char[]>(info.meshNameSize + 1);
			ifs.read(name.get(), info.meshNameSize);
			name[info.meshNameSize] = '\0';
			std::cout << name << std::endl;
		}

		{	// position, normals, tangents, bitangents
			std::vector<glm::vec3> vec3Attrib[TOTAL_VEC3_ATTRIBUTE];
			for (uint64_t i{}; i < TOTAL_VEC3_ATTRIBUTE; ++i)
				vec3Attrib[i].reserve( info.verticeCount );

			for (uint64_t i{}; i < TOTAL_VEC3_ATTRIBUTE; ++i)
			{
				std::unique_ptr<glm::vec3[]> ptr = std::make_unique<glm::vec3[]>(info.verticeCount);

				uint64_t const SIZE = sizeof(glm::vec3) * info.verticeCount;
				ifs.read(reinterpret_cast<char*>(ptr.get()), SIZE);

				vec3Attrib[i] = std::move( std::vector<glm::vec3>(ptr.get(), ptr.get() + info.verticeCount) );
			}
		}

		{	// texture coordinates, 
			std::vector<glm::vec2> vec2Attrib[TOTAL_VEC2_ATTRIBUTE];
			for (uint64_t i{}; i < TOTAL_VEC2_ATTRIBUTE; ++i)
				vec2Attrib[i].reserve(info.verticeCount);

			for (uint64_t i{}; i < TOTAL_VEC2_ATTRIBUTE; ++i)
			{
				std::unique_ptr<glm::vec2[]> ptr = std::make_unique<glm::vec2[]>(info.verticeCount);

				uint64_t const SIZE = sizeof(glm::vec2) * info.verticeCount;
				ifs.read(reinterpret_cast<char*>(ptr.get()), SIZE);

				vec2Attrib[i] = std::move( std::vector<glm::vec2>(ptr.get(), ptr.get() + info.verticeCount) );
			}
		}

		{	// indicies
			std::vector<uint32_t> indices{}; indices.reserve(info.indicesCount);
			ifs.read(reinterpret_cast<char*>(indices.data()), sizeof(uint32_t) * info.indicesCount);
		}

		{	// Material file paths
			std::string materialPaths[NUM_TEXTURE_TYPE]{};
			for (uint64_t i{}; i < NUM_TEXTURE_TYPE; ++i)
			{
				std::unique_ptr<char[]> path = std::make_unique<char[]>(info.materialPathSize[i] + 1);
				ifs.read(path.get(), info.materialPathSize[i]);
				path[info.materialPathSize[i]] = '\0';
				materialPaths[i] = path.get();
			}
		}
	}
}
