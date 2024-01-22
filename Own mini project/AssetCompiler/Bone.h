#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/scene.h>

#include <vector>

#include "Interpolation.h"

namespace Animation
{
	class Bone
	{
	private:
		glm::mat4 transform;
		std::vector<KeyPosition> positions;
		std::vector<KeyRotation> rotations;
		std::vector<KeyScale> scales;
		size_t numPositions;
		size_t numRotations;
		size_t numScalings;
		std::string name;
		unsigned int id;

	public:
		Bone(const std::string& inName, int inId, const aiNodeAnim* channel) 
		{
			name = inName;
			id = inId;
			transform = glm::mat4(1.0f);

			numPositions = channel->mNumPositionKeys;

			for (int positionIndex = 0; positionIndex < numPositions; ++positionIndex)
			{
				aiVector3D aiPosition = channel->mPositionKeys[positionIndex].mValue;
				float timeStamp = (float)channel->mPositionKeys[positionIndex].mTime;
				KeyPosition data = { glm::vec3(aiPosition.x, aiPosition.y, aiPosition.z), timeStamp };
				positions.push_back(data);
			}

			numRotations = channel->mNumRotationKeys;
			for (int rotationIndex = 0; rotationIndex < numRotations; ++rotationIndex)
			{
				aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
				float timeStamp = (float)channel->mRotationKeys[rotationIndex].mTime;
				KeyRotation data = { glm::quat(aiOrientation.w, aiOrientation.x, aiOrientation.y, aiOrientation.z), timeStamp };
				rotations.push_back(data);
			}

			numScalings = channel->mNumScalingKeys;
			for (int keyIndex = 0; keyIndex < numScalings; ++keyIndex)
			{
				aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
				float timeStamp = (float)channel->mScalingKeys[keyIndex].mTime;
				KeyScale data = { glm::vec3(scale.x, scale.y, scale.z), timeStamp };
				scales.push_back(data);
			}
		}

		std::vector<KeyPosition> const& GetKeyPositions(void) const { return positions; };
		std::vector<KeyRotation> const& GetKeyRotations(void) const { return rotations; };
		std::vector<KeyScale> const& GetKeyScale(void) const { return scales; };
	};
}