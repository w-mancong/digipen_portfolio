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
#include "meshoptimizer.h"
#include "GeomCompiler.h"

namespace
{
	aiTextureType constexpr const TEXTURE_TYPE[] = { aiTextureType_AMBIENT_OCCLUSION, aiTextureType_DIFFUSE, 
													 aiTextureType_METALNESS, aiTextureType_DIFFUSE_ROUGHNESS, 
													 aiTextureType_NORMALS };
	uint64_t constexpr const NUM_TEXTURE_TYPE = sizeof(TEXTURE_TYPE) / sizeof(*TEXTURE_TYPE);
}

std::ostream& operator<<(std::ostream& os, aiString s)
{
	return os << s.C_Str();
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

	// Optimizing

	// Binary export
	SerializationData data{};
	uint32_t const numMesh = m_Scene->mNumMeshes;
	for (uint32_t cnt{}; cnt < numMesh; ++cnt)
	{
		aiMesh* currMesh{ m_Scene->mMeshes[cnt] };
		Submesh submesh{};

		std::cout << currMesh->mName << std::endl;
		if (m_Scene->HasMaterials())
		{
			aiMaterial const* mat = m_Scene->mMaterials[currMesh->mMaterialIndex];
			for (uint64_t i{ 1 }; i <= 17; ++i)
			{
				//aiTextureType const type = *(TEXTURE_TYPE + i);
				aiTextureType const type = static_cast<aiTextureType>( i );
				uint64_t const count = mat->GetTextureCount(type);
				if (mat->GetTextureCount(type) > 0)
				{
					// Retrieve the first texture of the current type
					aiString texturePath{};
					if (AI_SUCCESS == mat->GetTexture(type, 0, &texturePath)) {
						std::cout << "Texture type: " << type << ", Path: " << texturePath.C_Str() << std::endl;
					}
					else {
						std::cerr << "Failed to retrieve texture path for type: " << type << std::endl;
					}
				}
			}
		}
	}



	//for (unsigned int i = 0; i < m_Scene->mNumMaterials; ++i) {
	//	aiMaterial* material = m_Scene->mMaterials[i];

	//	// Iterate through each texture type (diffuse, specular, normal, etc.)
	//	for (unsigned int j = 0; j < aiTextureType_UNKNOWN; ++j) {
	//		aiTextureType textureType = static_cast<aiTextureType>(j);

	//		// Check if the material has a texture of the current type
	//		if (material->GetTextureCount(textureType) > 0) 
	//		{
	//			// Retrieve the first texture of the current type
	//			aiString texturePath{};
	//			if (AI_SUCCESS == material->GetTexture(textureType, 0, &texturePath)) {
	//				std::cout << "Texture type: " << textureType << ", Path: " << texturePath.C_Str() << std::endl;
	//			}
	//			else {
	//				std::cerr << "Failed to retrieve texture path for type: " << textureType << std::endl;
	//			}
	//		}
	//	}
	//}

	// Exporting
	//std::string outputFile{ this->m_OutputFileDirectory + _inputFilepath.substr(_inputFilepath.find_last_of('\\'), _inputFilepath.find_last_of('.') - _inputFilepath.find_last_of('\\')) + ".h_mesh" };
	
	// Lib Exporting
	//try {
	//	if (m_Exporter.Export(this->m_Scene, this->m_ExportFileType, outputFile, m_ExportFlag) != AI_SUCCESS) {
	//		std::cout << ">> Error encountered during exporting\n";
	//		return false;
	//	}
	//}
	//catch (std::exception& _e) {
	//	std::cout << ">> Exception countered while exporting input data\n";
	//	std::cout << ">> Exception: " << _e.what() << std::endl;
	//	return false;
	//}

	// Self Binary Export
	//SerializationData data{};

	//for (unsigned int meshCount{ 0 }; meshCount < this->m_Scene->mNumMeshes; ++meshCount) {
	//	aiMesh* currMesh{ this->m_Scene->mMeshes[meshCount] };
	//	
	//	// Vertices, Normals, Tangents, Bitangents & TexCoords
	//	for (unsigned int vertexCount{ 0 }; vertexCount < currMesh->mNumVertices; ++vertexCount) {
	//		if (currMesh->HasPositions())
	//			data.vertices.emplace_back(ConvertaiVec3toVec3(currMesh->mVertices[vertexCount]));
	//
	//		if (currMesh->HasNormals())
	//			data.normals.emplace_back(ConvertaiVec3toVec3(currMesh->mNormals[vertexCount]));
	//
	//		if (currMesh->HasTangentsAndBitangents()) {
	//			data.tangents.emplace_back(ConvertaiVec3toVec3(currMesh->mTangents[vertexCount]));
	//			data.biTangents.emplace_back(ConvertaiVec3toVec3(currMesh->mBitangents[vertexCount]));
	//		}

	//		if (currMesh->HasTextureCoords(0))
	//			data.texCoords.emplace_back(ConvertaiVec3toVec2(currMesh->mTextureCoords[0][vertexCount]));
	//	}
	//	
	//	// Indices
	//	for (unsigned int faceCount{ 0 }; faceCount < currMesh->mNumFaces; ++faceCount) {
	//		aiFace currFace{ currMesh->mFaces[faceCount] };
	//		std::vector<unsigned int> faceIndices{};
	//		for (unsigned int indexCount{ 0 }; indexCount < currFace.mNumIndices; ++indexCount)
	//			faceIndices.emplace_back(currFace.mIndices[indexCount]);
	//		data.indices.emplace_back(faceIndices);
	//	}
	//}

	//std::cout << ">> Output file at " << outputFile << "\n";
	//if (Serialize(outputFile, data) == false)
	//	return false;

	//std::string testFile{ _inputFilepath.substr(0, _inputFilepath.find_last_of(".")) + ".h_test" };
	//SerializationData testData{};
	//Deserialize(outputFile,testData);
	//Serialize(testFile, testData);

	return true;
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

bool GeomCompiler::Serialize(const std::string& _filepath, const SerializationData& _data) {
	std::ofstream ofs{ _filepath, std::ios::out | std::ios::binary };
	if (ofs.is_open() == false) {
		std::cout << ">> Error encountered during serializiation. Output file could not be created\n";
		return false;
	}

	//ofs << "Vertices\n";
	//for (int i{ 0 }; i < _data.vertices.size(); ++i) {
	//	ofs << _data.vertices[i];
	//	if (i + 1 < _data.vertices.size())
	//		ofs << ",\n";
	//}

	//ofs << "\nNormals\n";
	//for (int i{ 0 }; i < _data.normals.size(); ++i) {
	//	ofs << _data.normals[i];
	//	if (i + 1 < _data.normals.size())
	//		ofs << ",\n";
	//}

	//ofs << "\nTangents\n";
	//for (int i{ 0 }; i < _data.tangents.size(); ++i) {
	//	ofs << _data.tangents[i];
	//	if (i + 1 < _data.tangents.size())
	//		ofs << ",\n";
	//}

	//ofs << "\nBiTangents\n";
	//for (int i{ 0 }; i < _data.biTangents.size(); ++i) {
	//	ofs << _data.biTangents[i];
	//	if (i + 1 < _data.biTangents.size())
	//		ofs << ",\n";
	//}

	//ofs << "\nTexCoords\n";
	//for (int i{ 0 }; i < _data.texCoords.size(); ++i) {
	//	ofs << _data.texCoords[i];
	//	if (i + 1 < _data.texCoords.size())
	//		ofs << ",\n";
	//}

	//ofs << "\nIndices\n";
	//for (int i{ 0 }; i < _data.indices.size(); ++i) {
	//	for (unsigned int j{ 0 }; j < _data.indices[i].size(); ++j) {
	//		ofs << _data.indices[i][j];
	//		if (j + 1 < _data.indices[i].size())
	//			ofs << ",";
	//	}
	//	if (i + 1 < _data.indices.size())
	//		ofs << ",\n";
	//}

	ofs.close();
	return true;
}

template <typename Out>
void split(const std::string& s, char delim, Out result) {
	std::istringstream iss(s);
	std::string item;
	while (std::getline(iss, item, delim)) {
		*result++ = item;
	}
}

std::vector<std::string> split(const std::string& _string, char _delimiter) {
	std::vector<std::string> result;
	split(_string, _delimiter, std::back_inserter(result));
	return result;
}

bool GeomCompiler::Deserialize(const std::string& _filepath, SerializationData& _data) {
	//return Deserialize(_filepath, _data.vertices, _data.normals, _data.tangents, _data.biTangents, _data.texCoords, _data.indices);
	return false;
}

bool GeomCompiler::Deserialize(const std::string& _filepath,
								std::vector<glm::vec3>& _vertices,
								std::vector<glm::vec3>& _normals,
								std::vector<glm::vec3>& _tangents,
								std::vector<glm::vec3>& _biTangents,
								std::vector<glm::vec2>& _texCoords,
								std::vector<std::vector<unsigned int>>& _indices) {
	std::ifstream ifs{ _filepath, std::ios::in | std::ios::binary };
	if (ifs.is_open() == false) {
		std::cout << ">> Error encountered during deserialization. Input file could not be opened\n";
		return false;
	}

	_vertices.clear();
	_normals.clear();
	_tangents.clear();
	_biTangents.clear();
	_texCoords.clear();
	_indices.clear();

	std::string currLine{};
	ifs >> currLine;

	if (currLine == "Vertices") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			if (result.size() == 3)
				_vertices.emplace_back(glm::vec3{ std::stof(result[0]), std::stof(result[1]), std::stof(result[2]) });
			else
				break;
		} while (ifs >> currLine);
	}

	if (currLine == "Normals") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			if (result.size() == 3)
				_normals.emplace_back(glm::vec3{ std::stof(result[0]), std::stof(result[1]), std::stof(result[2]) });
			else
				break;
		} while (ifs >> currLine);
	}

	if (currLine == "Tangents") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			if (result.size() == 3)
				_tangents.emplace_back(glm::vec3{ std::stof(result[0]), std::stof(result[1]), std::stof(result[2]) });
			else
				break;
		} while (ifs >> currLine);
	}

	if (currLine == "BiTangents") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			if (result.size() == 3)
				_biTangents.emplace_back(glm::vec3{ std::stof(result[0]), std::stof(result[1]), std::stof(result[2]) });
			else
				break;
		} while (ifs >> currLine);
	}

	if (currLine == "TexCoords") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			if (result.size() == 2)
				_texCoords.emplace_back(glm::vec2{ std::stof(result[0]), std::stof(result[1]) });
			else
				break;
		} while (ifs >> currLine);
	}

	if (currLine == "Indices") {
		ifs >> currLine;
		do {
			std::vector<std::string> result{ split(currLine, ',') };
			std::vector<unsigned int> indices{};
			for (size_t i{ 0 }; i < result.size(); ++i)
				indices.emplace_back(std::stoul(result[i]));
			_indices.emplace_back(indices);
		} while (ifs >> currLine);
	}



	ifs.close();
	return true;
}