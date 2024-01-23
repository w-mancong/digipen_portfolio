/********************************************************************************************************************************************************
#    All content © 2023 DigiPen Institute of Technology Singapore, all rights reserved.
#    Academic Year:			Trimester 1 Fall 2023
#    Team Name:				Horroscope
#    Game Name:				The Vessel
#    Module:				CSD3400
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#    File Name:				GeomCompiler.cpp
#    Primary Author:		Wong Man Cong
#    Secondary Author:		-
*********************************************************************************************************************************************************/

// --------------------
// Header Includes
// --------------------
#include <iostream>
#include <filesystem>
#include <fstream>
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

glm::mat4 ConvertaiMat4toMat4(aiMatrix4x4 const& mat)
{
	return
	{
		{ mat.a1, mat.a2, mat.a3, mat.a4 },
		{ mat.b1, mat.b2, mat.b3, mat.b4 },
		{ mat.c1, mat.c2, mat.c3, mat.c4 },
		{ mat.d1, mat.d2, mat.d3, mat.d4 },
	};
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

		if (m_Scene && m_Scene->mRootNode) {
			// Create a scaling matrix
			aiMatrix4x4 scalingMatrix;
			aiMatrix4x4::Scaling(aiVector3D(0.01f, 0.01f, 0.01f), scalingMatrix);

			// Apply the scaling matrix to the root node
			m_Scene->mRootNode->mTransformation = scalingMatrix * m_Scene->mRootNode->mTransformation;
		}
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

	CompiledMesh data{}; AnimationData aniData{};
	ProcessNode(m_Scene->mRootNode, data);
	ProcessAnimation(data, aniData);

	// Exporting
	//std::string const outputFile{ this->m_OutputFileDirectory + _inputFilepath.substr(_inputFilepath.find_last_of('\\'), _inputFilepath.find_last_of('.') - _inputFilepath.find_last_of('\\')) + ".h_mesh" };
	std::string const outputFile{ _inputFilepath.substr(0, _inputFilepath.find_last_of('.')) + ".h_mesh" };
	return Deserialize(outputFile, data);
}

void GeomCompiler::ProcessNode(aiNode* node, CompiledMesh& data, const aiMatrix4x4& parentTransform) const
{
	// Binary export
	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	{	// Process all the node's meshes (if any)
		aiMesh* currMesh = m_Scene->mMeshes[node->mMeshes[i]];
		Submesh submesh{};

		submesh.meshName = currMesh->mName.C_Str();
		if (m_Scene->HasMaterials())
			LoadMaterial(submesh, currMesh);

		LoadVertices(submesh, currMesh);
		LoadIndices(submesh, currMesh);
		ExtractBoneWeightForVertices(submesh.vertices, data.boneProps, currMesh);

		// Optimize mesh base on it's loaded vertices and indices
		OptimizeMesh(submesh);

		// Update the transform matrix based on root node
		aiMatrix4x4 currentTransform = parentTransform * node->mTransformation;
		submesh.transformMatrix = ConvertaiMat4toMat4(currentTransform.Transpose());
		data.meshInfos.emplace_back(submesh);
	}
	// Then do the same for each of its children
	for (uint32_t i = 0; i < node->mNumChildren; i++)
		ProcessNode(node->mChildren[i], data, parentTransform * node->mTransformation);
}

void GeomCompiler::LoadMaterial(Submesh& submesh, aiMesh const* currMesh) const
{
	aiMaterial const* mat = m_Scene->mMaterials[currMesh->mMaterialIndex];
	submesh.materialName = mat->GetName().C_Str();

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
}

void GeomCompiler::LoadVertices(Submesh& submesh, aiMesh const* currMesh) const
{
	uint32_t const numVertices = currMesh->mNumVertices;
	for (uint32_t vertIdx{}; vertIdx < numVertices; ++vertIdx)
	{
		Vertex vertex{};
		if (currMesh->HasPositions())
			vertex.position = ConvertaiVec3toVec3(currMesh->mVertices[vertIdx]);
		if (currMesh->HasNormals())
			vertex.normal = ConvertaiVec3toVec3(currMesh->mNormals[vertIdx]);
		if (currMesh->HasTangentsAndBitangents())
		{
			vertex.tangent = ConvertaiVec3toVec3(currMesh->mTangents[vertIdx]);
			vertex.bitangent = ConvertaiVec3toVec3(currMesh->mBitangents[vertIdx]);
		}
		if (currMesh->HasTextureCoords(0))
			vertex.uv = ConvertaiVec3toVec2(currMesh->mTextureCoords[0][vertIdx]);
		submesh.vertices.emplace_back(vertex);
	}
}

void GeomCompiler::LoadIndices(Submesh& submesh, aiMesh const* currMesh) const
{
	uint32_t const numFaces = currMesh->mNumFaces;
	for (uint32_t faceIdx{}; faceIdx < numFaces; ++faceIdx)
	{
		aiFace const currFace{ currMesh->mFaces[faceIdx] };
		uint32_t const indicesCnt = currFace.mNumIndices;
		for (uint32_t indicesIdx{}; indicesIdx < indicesCnt; ++indicesIdx)
			submesh.indices.emplace_back(currFace.mIndices[indicesIdx]);
	}
}

void GeomCompiler::OptimizeMesh(MC::Submesh& submesh) const
{
	uint64_t const indices_counter = submesh.indices.size();
	std::vector<uint32_t> remap(indices_counter);

	uint64_t const vertex_counter = meshopt_generateVertexRemap(&remap[0], submesh.indices.data(), indices_counter, submesh.vertices.data(), indices_counter, sizeof(Vertex));
	Submesh opMesh{};
	opMesh.indices.resize(indices_counter);
	opMesh.vertices.resize(vertex_counter);

	meshopt_remapIndexBuffer(opMesh.indices.data(), submesh.indices.data(), indices_counter, &remap[0]);

	meshopt_optimizeVertexCache(opMesh.indices.data(), opMesh.indices.data(), indices_counter, vertex_counter);

	meshopt_remapVertexBuffer(opMesh.vertices.data(), submesh.vertices.data(), indices_counter, sizeof(Vertex), &remap[0]);

	meshopt_optimizeOverdraw(opMesh.indices.data(), opMesh.indices.data(), indices_counter, &opMesh.vertices[0].position.x, vertex_counter, sizeof(Vertex), 1.05f);

	opMesh.vertices.resize(meshopt_optimizeVertexFetch(opMesh.vertices.data(), opMesh.indices.data(), indices_counter, opMesh.vertices.data(), vertex_counter, sizeof(Vertex)));

	submesh.indices = std::exchange(opMesh.indices, {});
	submesh.vertices = std::exchange(opMesh.vertices, {});
}

void GeomCompiler::ExtractBoneWeightForVertices(std::vector<Vertex>& vertices, std::vector<Ani::BoneProps>& boneProps, aiMesh const* const mesh) const
{
	// Set the maximum bones to 100
	unsigned numBones = mesh->mNumBones > 100 ? 100 : mesh->mNumBones;
	// For each bone
	for (unsigned boneIndex{}; boneIndex < numBones; ++boneIndex)
	{
		int boneID = -1;
		char const* boneName = mesh->mBones[boneIndex]->mName.C_Str();
		if (boneIndex >= boneProps.size())
		{
			boneProps.push_back( { boneName, ConvertaiMat4toMat4(mesh->mBones[boneIndex]->mOffsetMatrix) } );
			boneID = boneIndex;
		}
		else
		{
			for (unsigned i{}; i < boneProps.size(); ++i)
			{
				if (boneProps[i].name == boneName)
				{
					boneID = i;
					break;
				}
			}
		}
		assert(boneID != -1);

		// Get all vertex weights for current bone
		aiVertexWeight* weights = mesh->mBones[boneIndex]->mWeights;
		unsigned int numWeights = mesh->mBones[boneIndex]->mNumWeights;

		// For each weight at vertex x for current bone
		for (int weightIndex{}; weightIndex < numWeights; ++weightIndex)
		{
			unsigned vertexID = weights[weightIndex].mVertexId;
			float weight = weights[weightIndex].mWeight;
			assert(vertexID <= vertices.size());

			// Update four most influential bones
			for (int i{}; i < 4; ++i)
			{
				if (vertices[vertexID].boneIDs[i] < 0)
				{
					vertices[vertexID].weights[i] = weight;
					vertices[vertexID].boneIDs[i] = boneID;
					break;
				}
			}
		}
	}
}

void GeomCompiler::ProcessAnimation(MC::CompiledMesh& data, AnimationData& aniData) const
{
	if (m_Scene->mNumAnimations <= 0)
		return;

	aiAnimation* animation = *m_Scene->mAnimations;
	aniData.duration = static_cast<float>(animation->mDuration);
	aniData.tps = static_cast<float>(animation->mTicksPerSecond);
	GenerateBoneTree(&aniData.rootNode, m_Scene->mRootNode, aniData.boneData);
	// Reset all root transformations
	aniData.rootNode.transformation = glm::mat4(1.0f);
	LoadIntermediateBones(animation, data.boneProps, aniData);
	// serialize animation data
	aniData.clipName = animation->mName.data;
	DeserializeAnimation(aniData);
}

void GeomCompiler::GenerateBoneTree(Ani::AssimpNodeData* parent, aiNode const* src, BoneTreeData& boneData) const
{
	assert(src);

	parent->name = src->mName.data;
	parent->transformation = ConvertaiMat4toMat4(src->mTransformation);
	parent->childrenCount = src->mNumChildren;
	// Bone data
	boneData.assimpNodeDataChildrenCount.emplace_back(parent->childrenCount);
	boneData.assimpNodeData.emplace_back(BoneTreeData::NodeData( parent->transformation, parent->name ));

	for (uint32_t i{}; i < parent->childrenCount; ++i)
	{
		Ani::AssimpNodeData newData{};
		GenerateBoneTree(&newData, src->mChildren[i], boneData);
		parent->children.emplace_back(newData);
	}
}

void GeomCompiler::LoadIntermediateBones(aiAnimation const* animation, std::vector<Animation::BoneProps>& boneProps, MC::AnimationData& aniData) const
{
	for (int i{}; i < animation->mNumChannels; ++i)
	{
		aiNodeAnim* channel = animation->mChannels[i];
		char const* boneName = channel->mNodeName.data;
		int boneID = -1;

		for (uint32_t j{}; j < boneProps.size(); ++j)
		{
			if (!strcmp(boneProps[i].name.c_str(), boneName))
			{
				boneID = i;
				break;
			}
		}

		if (boneProps.size() < 100)
		{
			if (boneID == -1)
			{
				Ani::BoneProps boneProp{};
				boneProp.name = boneName;
				boneProps.emplace_back(boneProp);
				boneID = boneProps.size() - 1;
			}
		}
		aniData.bones.emplace_back(Ani::Bone(channel->mNodeName.data, boneID, channel));
	}
	aniData.boneProps = boneProps;
}

void GeomCompiler::DeserializeAnimation(MC::AnimationData const& aniData) const
{
	std::string path_to_check = "Assets\\Animation";
	if (!std::filesystem::exists(path_to_check))
		std::filesystem::create_directories(path_to_check);
	std::ofstream ofs{ path_to_check + "\\" + aniData.clipName + ".h_anim", std::ios::out | std::ios::binary };
	if (!ofs.is_open())
	{
		std::cout << ">> Error encountered during deserialization of animation data. Output file could not be created\n";
		return;
	}

	AnimationHeaderInfo const info
	{
		.clipNameSize		= aniData.clipName.size(),
		.bonesSize			= aniData.bones.size(),
		.bonePropsSize		= aniData.boneProps.size(),
		.assimpNodeDataSize = aniData.boneData.assimpNodeDataChildrenCount.size()
	};

	// Write the offset for the animation data
	ofs.write(reinterpret_cast<char const*>(&info), sizeof(AnimationHeaderInfo));

	// Write the name of the clip name
	ofs.write(aniData.clipName.c_str(), info.clipNameSize);

	// Write bone data
	ofs.write(reinterpret_cast<char const*>(aniData.bones.data()), sizeof(Ani::Bone) * info.bonesSize);

	// Write bone props data
	ofs.write(reinterpret_cast<char const*>(aniData.boneProps.data()), sizeof(Ani::BoneProps) * info.bonePropsSize);

	// Write animation's tick per second
	ofs.write(reinterpret_cast<char const*>(&aniData.tps), sizeof(float));

	// Write animation's duration
	ofs.write(reinterpret_cast<char const*>(&aniData.duration), sizeof(float));

	// Write each assimp data node's children count
	ofs.write(reinterpret_cast<char const*>(aniData.boneData.assimpNodeDataChildrenCount.data()), sizeof(uint32_t) * info.assimpNodeDataSize);

	// Write each assimp data node's transformation and name data
	ofs.write(reinterpret_cast<char const*>(aniData.boneData.assimpNodeData.data()), sizeof(BoneTreeData::NodeData) * info.assimpNodeDataSize);
}

bool GeomCompiler::Deserialize(std::string const& outputFile, CompiledMesh const& data)
{
	std::ofstream ofs{ outputFile, std::ios::out | std::ios::binary };
	if (!ofs.is_open()) {
		std::cout << ">> Error encountered during deserializiation. Output file could not be created\n";
		return false;
	}

	uint64_t const numMesh = data.meshInfos.size();
	// Write the numebr of submeshes into file
	ofs.write(reinterpret_cast<char const*>(&numMesh), sizeof(uint64_t));
	{
		uint64_t const bonePropsSize = data.boneProps.size();
		// Write the size of bone props
		ofs.write(reinterpret_cast<char const*>(&bonePropsSize), sizeof(uint64_t));
		// Write the data of bone props into file
		ofs.write(reinterpret_cast<char const*>(data.boneProps.data()), sizeof(Ani::BoneProps) * bonePropsSize);
	}


	for (uint64_t meshIdx{}; meshIdx < numMesh; ++meshIdx)
	{
		Submesh const& submesh = data.meshInfos[meshIdx];

		HeaderInfo const info
		{
			.meshNameSize = submesh.meshName.size(),
			.verticeCount = submesh.vertices.size(),
			.indicesCount = submesh.indices.size(),
			.materialNameSize = submesh.materialName.size(),
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

		// Write the data of vertices
		ofs.write(reinterpret_cast<char const*>(submesh.vertices.data()), sizeof(Vertex) * info.verticeCount);

		// Write indices
		ofs.write(reinterpret_cast<char const*>(submesh.indices.data()), sizeof(uint32_t) * info.indicesCount);

		// Material name
		ofs.write(submesh.materialName.c_str(), info.materialNameSize);

		// Write string of material path
		for (uint64_t i{}; i < NUM_TEXTURE_TYPE; ++i)
			ofs.write(submesh.materialPaths[i].c_str(), info.materialPathSize[i]);

		// Write model matrix
		ofs.write(reinterpret_cast<char const*>(&submesh.transformMatrix[0][0]), sizeof(glm::mat4));
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
	{
		uint64_t bonePropsSize{};
		// Read the total number of bone props
		ifs.read(reinterpret_cast<char*>(&bonePropsSize), sizeof(uint64_t));
		// Read bone props data
		std::vector<Ani::BoneProps> boneProps{}; boneProps.reserve(bonePropsSize);
		ifs.read(reinterpret_cast<char*>(boneProps.data()), sizeof(Ani::BoneProps) * bonePropsSize);
	}

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

		{	// Vertices
			std::unique_ptr<Vertex[]> vertices = std::make_unique<Vertex[]>(info.verticeCount);
			ifs.read(reinterpret_cast<char*>(vertices.get()), sizeof(Vertex) * info.verticeCount);
		}

		{	// indicies
			std::vector<uint32_t> indices{}; indices.reserve(info.indicesCount);
			ifs.read(reinterpret_cast<char*>(indices.data()), sizeof(uint32_t) * info.indicesCount);
		}

		{	// Material name
			std::unique_ptr<char[]> name = std::make_unique<char[]>(info.materialNameSize + 1);
			ifs.read(name.get(), info.materialNameSize);
			name[info.materialNameSize] = '\0';
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

		{	// Read transform matrix
			glm::mat4 mat(1.0f);
			ifs.read(reinterpret_cast<char*>(&mat[0][0]), sizeof(glm::mat4));
		}
	}
}
