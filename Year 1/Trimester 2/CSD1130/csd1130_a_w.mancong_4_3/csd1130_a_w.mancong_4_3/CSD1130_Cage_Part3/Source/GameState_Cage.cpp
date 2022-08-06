/*!*****************************************************************************
\file		GameState_Cage.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	04-04-2022
\brief
This file contain functions definition for GameStateCage to
Load, Init, Update, Draw, Free and Unload relavent behaviour for this game state

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*******************************************************************************/

#include "main.h"

/******************************************************************************/
/*!
	Defines
*/
/******************************************************************************/
const unsigned int	GAME_OBJ_NUM_MAX		= 32;	//The total number of different objects (Shapes)
const unsigned int	GAME_OBJ_INST_NUM_MAX	= 2048;	//The total number of different game object instances

//Flags
const unsigned int	FLAG_ACTIVE				= 0x00000001;
const unsigned int	FLAG_VISIBLE			= 0x00000002;
const unsigned int	FLAG_NON_COLLIDABLE		= 0x00000004;

const float			PI_OVER_180				= PI / 180.0f;

//values: 0,1,2,3
//0: original: 
//1: Extra Credits: with line edges only 
//2: Extra Credits: circle-circle-with-mass only (no line edges)
//3: 1&2 = All extra credits included
int EXTRA_CREDITS = 3;

enum class TYPE_OBJECT
{
	TYPE_OBJECT_BALL,	//0
	TYPE_OBJECT_WALL,	//1
	TYPE_OBJECT_PILLAR, //2

	TYPE_OBJECT_NUM
};

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/
struct GameObj
{
	TYPE_OBJECT	type;		// object type
	AEGfxVertexList *	pMesh;		// pbject
};

struct GameObjInst
{
	GameObj*		pObject;	// pointer to the 'original'
	unsigned int	flag;		// bit flag or-ed together
	float			scale;
	CSD1130::Vec2	posCurr;	// object current position
	CSD1130::Vec2	velCurr;	// object current velocity
	float			dirCurr;	// object current direction
	float			speed;

	CSD1130::Mtx33	transform;	// object drawing matrix

	// pointer to custom data specific for each object type
	void*			pUserData;
};


/******************************************************************************/
/*!
	File globals
*/
/******************************************************************************/
// list of original objects
static GameObj			*sGameObjList;
static unsigned int		sGameObjNum;

// list of object instances
static GameObjInst		*sGameObjInstList;
static unsigned int		sGameObjInstNum;

/*!*****************************************************************************
\brief
	Create and instance of game object of the specified type
\param [in] type:
	Type of game object to be created
\param [in] scale:
	Size of the game object to be created
\param [in] pPos:
	Initial position of game object to be created
\param [in] pVel:
	Initial velocity of game object to be created
\param [in] dir:
	Initial direction of how much the game object will be rotated when it is
	created
*******************************************************************************/
GameObjInst*		gameObjInstCreate (TYPE_OBJECT type,
										   float scale, 
										   CSD1130::Vec2* pPos,
										   CSD1130::Vec2* pVel,
										   float dir);

/*!*****************************************************************************
\brief
	Set the active state of this game object instance to false
\param [out] pInst:
	Game object instance to set it's active flag to false
*******************************************************************************/
void				gameObjInstDestroy(GameObjInst* pInst);

static Circle			*sBallData = 0;
static LineSegment		*sWallData = 0;
static Circle			*sPillarData = 0;

/*!*****************************************************************************
\brief
	Load function for GameStateCage
*******************************************************************************/
void GameStateCageLoad(void)
{
	sGameObjList = (GameObj *)calloc(GAME_OBJ_NUM_MAX, sizeof(GameObj));
	sGameObjInstList = (GameObjInst *)calloc(GAME_OBJ_INST_NUM_MAX, sizeof(GameObjInst));
	sGameObjNum = 0;

	GameObj* pObj; int Parts;

	//------------------------------------------

	//Creating the ball object
	pObj		= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type = TYPE_OBJECT::TYPE_OBJECT_BALL;

		AEGfxMeshStart();

		//1st argument: X
		//2nd argument: Y
		//3rd argument: ARGB

		//Creating the ball shape
		Parts = 36;
		for (float i = 0; i < Parts; ++i)
		{
			AEGfxTriAdd(
				0.0f, 0.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf(i * 2 * PI / Parts) * 1.0f, sinf(i * 2 * PI / Parts) * 1.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf((i + 1) * 2 * PI / Parts) * 1.0f, sinf((i + 1) * 2 * PI / Parts) * 1.0f, 0xFFFFFF00, 0.0f, 0.0f);
		}

		pObj->pMesh = AEGfxMeshEnd();
	}
	//------------------------------------------

	// Creating the wall object
	pObj		= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type = TYPE_OBJECT::TYPE_OBJECT_WALL;

		AEGfxMeshStart();

		//1st argument: X
		//2nd argument: Y
		//3rd argument: ARGB

		//Creating the wall shape
		AEGfxVertexAdd(-0.5f, 0.0f, 0xFFFF0000, 0.0f, 0.0f);
		AEGfxVertexAdd(0.5f, 0.0f, 0xFFFFFFFF, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
	}
	//------------------------------------------

	//Creating the pillar object
	pObj		= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type = TYPE_OBJECT::TYPE_OBJECT_PILLAR;

		AEGfxMeshStart();

		//1st argument: X
		//2nd argument: Y
		//3rd argument: ARGB

		//Creating the pillar shape
		Parts = 24;
		for (float i = 0; i < Parts; ++i)
		{
			AEGfxTriAdd(
				0.0f, 0.0f, 0xFFFF2211, 0.0f, 0.0f,
				cosf(i * 2 * PI / Parts) * 1.0f, sinf(i * 2 * PI / Parts) * 1.0f, 0xFFFFFFFF, 0.0f, 0.0f,
				cosf((i + 1) * 2 * PI / Parts) * 1.0f, sinf((i + 1) * 2 * PI / Parts) * 1.0f, 0xFFFFFF11, 0.0f, 0.0f);
		}

		pObj->pMesh = AEGfxMeshEnd();
	}

	AEGfxSetBackgroundColor(0.2f, 0.2f, 0.2f);	
}

/*!*****************************************************************************
\brief
	Init function for GameStateCage
*******************************************************************************/
void GameStateCageInit(void)
{
	GameObjInst *pInst;
	std::string str;
	std::ifstream inFile;

	if(EXTRA_CREDITS == 0)
		inFile.open("..\\Bin\\Resources\\LevelData - Original.txt");
	else if (EXTRA_CREDITS == 1)
		inFile.open("..\\Bin\\Resources\\LevelData - Extra - NoMass.txt");
	else if(EXTRA_CREDITS == 2)
		inFile.open("..\\Bin\\Resources\\LevelData - Extra - WithMass1.txt");
	else// if(EXTRA_CREDITS == 3)
		inFile.open("..\\Bin\\Resources\\LevelData - Extra - WithMass2.txt");
	
	if(inFile.is_open())
	{
		// read ball data
		float dir, speed, scale;
		unsigned int ballNum = 0;
		inFile >> ballNum;
		sBallData = new Circle[ballNum];

		for(unsigned int i = 0; i < ballNum; ++i)
		{
			// read pos
			inFile >> str >> sBallData[i].m_center.x;
			inFile >> str >> sBallData[i].m_center.y;
			// read direction
			inFile >> str >> dir;
			// read speed
			inFile >> str >> speed;
			// read radius
			inFile >> str >> sBallData[i].m_radius;
			if (EXTRA_CREDITS >= 2)
			{
				// read radius
				inFile >> str >> sBallData[i].m_mass;
			}
			// create ball instance
			CSD1130::Vec2 vel{ cosf(dir * PI_OVER_180) * speed, sinf(dir * PI_OVER_180) * speed };
			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_BALL, sBallData[i].m_radius,
										&sBallData[i].m_center, &vel, 0.0f);
			AE_ASSERT(pInst);
			pInst->speed = speed;
			pInst->pUserData = &sBallData[i];
		}

		// read wall data
		unsigned int wallNum = 0;
		CSD1130::Vec2 pos;

		inFile >> wallNum;
		sWallData = new LineSegment[wallNum];

		for(unsigned int i = 0; i < wallNum; ++i)
		{
			inFile >> str >> pos.x;
			inFile >> str >> pos.y;
			inFile >> str >> dir;
			inFile >> str >> scale;
			BuildLineSegment(sWallData[i], pos, scale, dir * PI_OVER_180);
			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_WALL, scale, &pos, 0, dir * PI_OVER_180);
			AE_ASSERT(pInst);
			pInst->pUserData = &sWallData[i];
		}

		// read pillar data
		unsigned int pillarNum = 0;
		
		inFile >> pillarNum;
		sPillarData = new Circle[pillarNum];

		for(unsigned int i = 0; i < pillarNum; ++i)
		{
			// read pos
			inFile >> str >> sPillarData[i].m_center.x;
			inFile >> str >> sPillarData[i].m_center.y;
			// read radius
			inFile >> str >> sPillarData[i].m_radius;
			// create pillar instance
			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_PILLAR, sPillarData[i].m_radius,
										&sPillarData[i].m_center, 0, 0.0f);
			AE_ASSERT(pInst);
			pInst->speed = 0.0f;
			pInst->pUserData = &sPillarData[i];
		}

		inFile.clear();
		inFile.close();
	}
	else
	{
		//AE_ASSERT_MESG(inFile, "Failed to open the text file");
		printf("Failed to open the text file");
	}
}

/*!*****************************************************************************
\brief
	Update function for GameStateCage
*******************************************************************************/
void GameStateCageUpdate(void)
{
	static bool full_screen_me;
	if (AEInputCheckTriggered(AEVK_F))
	{
		full_screen_me = !full_screen_me;
		AEToogleFullScreen(full_screen_me);
	}
	
	CSD1130::Vec2	interPtA, interPtB;
	CSD1130::Vec2   normalAtCollision;
	float			interTime = 0.0f;

	//f32 fpsT = (f32)AEFrameRateControllerGetFrameTime();

	//Update object instances positions
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst *pBallInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pBallInst->flag & FLAG_ACTIVE) ||
			pBallInst->pObject->type != TYPE_OBJECT::TYPE_OBJECT_BALL)
			continue;

		CSD1130::Vec2 posNext;
		posNext.x = pBallInst->posCurr.x + pBallInst->velCurr.x * g_dt;
		posNext.y = pBallInst->posCurr.y + pBallInst->velCurr.y * g_dt;

		// Update the latest ball data with the lastest ball's position
		Circle &ballData = *((Circle*)pBallInst->pUserData);
		ballData.m_center.x = pBallInst->posCurr.x;
		ballData.m_center.y = pBallInst->posCurr.y;

		// Check collision with walls
		for(unsigned int j = 0; j < GAME_OBJ_INST_NUM_MAX; ++j)
		{
			GameObjInst *pInst = sGameObjInstList + j;

			if (0 == (pInst->flag & FLAG_ACTIVE))
				continue;

			switch(pInst->pObject->type)
			{
				case TYPE_OBJECT::TYPE_OBJECT_WALL:
				{
					LineSegment &lineSegData = *((LineSegment*)pInst->pUserData);

					bool checkLineEdges = true;
					if (EXTRA_CREDITS == 2)
						checkLineEdges = false;

					if(CollisionIntersection_CircleLineSegment(ballData, 
																   posNext, 
																   lineSegData, 
																   interPtA,
																   normalAtCollision,
																   interTime,
																   checkLineEdges))
					{
						CSD1130::Vec2 reflectedVec;

						CollisionResponse_CircleLineSegment(interPtA, 
															    normalAtCollision,
																posNext, 
																reflectedVec);

						pBallInst->velCurr.x = reflectedVec.x * pBallInst->speed;
						pBallInst->velCurr.y = reflectedVec.y * pBallInst->speed;
					}
				}
				break;
				
				case TYPE_OBJECT::TYPE_OBJECT_PILLAR:
				{
					Circle &pillarData = *((Circle*)pInst->pUserData);

					CSD1130::Vec2 velA, velB;
					velA = pBallInst->velCurr * g_dt, velB = pInst->velCurr * g_dt;
					
					if(CollisionIntersection_CircleCircle(ballData, velA, pillarData, 
														velB, interPtA, interPtB, interTime))
					{
						CSD1130::Vec2 normal;
						normal = interPtA - pInst->posCurr;
						CSD1130::Vector2DNormalize(normal, normal);

						CSD1130::Vec2 reflectedVecNor;
						CollisionResponse_CirclePillar(normal, interTime, pBallInst->posCurr, interPtA,
													   posNext, reflectedVecNor);

						pBallInst->velCurr.x = reflectedVecNor.x * pBallInst->speed;
						pBallInst->velCurr.y = reflectedVecNor.y * pBallInst->speed;
					}
				}
				break;

				case TYPE_OBJECT::TYPE_OBJECT_BALL:
				{
					if ((EXTRA_CREDITS < 2) || (pInst == pBallInst))
						continue;

					Circle &otherBallData = *((Circle*)pInst->pUserData);

					CSD1130::Vec2 velA, velB;
					velA = pBallInst->velCurr * g_dt, velB = pInst->velCurr * g_dt;

					if (CollisionIntersection_CircleCircle(ballData, velA, otherBallData,
						velB, interPtA, interPtB, interTime))
					{

						if (EXTRA_CREDITS >= 2)
						{
							CSD1130::Vec2 reflectedVecA, reflectedVecB;
							CSD1130::Vec2 posNextB;//not used yet, even though computed below

							CSD1130::Vec2 normal;
							normal = interPtA - interPtB;
							CSD1130::Vector2DNormalize(normal, normal);

							CollisionResponse_CircleCircle(normal, interTime, velA, ballData.m_mass, interPtA, velB, otherBallData.m_mass, interPtB,
								reflectedVecA, posNext, reflectedVecB, posNextB);

							pBallInst->speed = CSD1130::Vector2DLength(reflectedVecA) / g_dt; //A: new speed
							CSD1130::Vector2DNormalize(reflectedVecA, reflectedVecA); //A: new speed direction
							
							pBallInst->velCurr.x = reflectedVecA.x * pBallInst->speed;
							pBallInst->velCurr.y = reflectedVecA.y * pBallInst->speed;

							pInst->speed = CSD1130::Vector2DLength(reflectedVecB) / g_dt; //B: new speed
							CSD1130::Vector2DNormalize(reflectedVecB, reflectedVecB); //B: new speed direction

							pInst->velCurr.x = reflectedVecB.x * pInst->speed;
							pInst->velCurr.y = reflectedVecB.y * pInst->speed;
						}
						//else if (EXTRA_CREDITS == 1)//this is internal only - circle-circle no mass
						//{

						//	CSD1130::Vec2 normal;
						//	normal = interPtA - pInst->posCurr;
						//	CSD1130::Vector2DNormalize(normal, normal);

						//	CSD1130::Vec2 reflectedVecNor;
						//	CollisionResponse_CirclePillar(normal, interTime, pBallInst->posCurr, interPtA,
						//		posNext, reflectedVecNor);

						//	pBallInst->velCurr.x = reflectedVecNor.x * pBallInst->speed;
						//	pBallInst->velCurr.y = reflectedVecNor.y * pBallInst->speed;

						//	//reflect the other ball
						//	CSD1130::Vec2 posNextB;//not used yet, even though computed below
						//	normal = -normal;
						//	CollisionResponse_CirclePillar(normal, interTime, pInst->posCurr, interPtA,
						//		posNextB, reflectedVecNor);

						//	pInst->velCurr.x = reflectedVecNor.x * pInst->speed;
						//	pInst->velCurr.y = reflectedVecNor.y * pInst->speed;
						//}
					}
					break;
				}
			}
		}

		pBallInst->posCurr.x = posNext.x;
		pBallInst->posCurr.y = posNext.y;
	}
	
	//Computing the transformation matrices of the game object instances
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		CSD1130::Mtx33 scale, rot, trans;
		GameObjInst *pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		CSD1130::Mtx33Scale(scale, pInst->scale, pInst->scale);
		CSD1130::Mtx33RotRad(rot, pInst->dirCurr);
		CSD1130::Mtx33Translate(trans, pInst->posCurr.x, pInst->posCurr.y);

		pInst->transform = trans * (rot * scale);
	}

	if(AEInputCheckTriggered(AEVK_R))
		gGameStateNext = GS_STATE::GS_RESTART;
}

/*!*****************************************************************************
\brief
	Render function for GameStateCage
*******************************************************************************/
void GameStateCageDraw(void)
{
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxTextureSet(NULL, 0, 0);
	AEGfxSetTransparency(1.0f);

	
	//Drawing the object instances
	int only4 = 0;
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE) || 0 == (pInst->flag & FLAG_VISIBLE))
			continue;
		
		f32 t[3][3]; size_t k = 0;
		for (size_t l = 0; l < 3; ++l)
			for (size_t j = 0; j < 3; ++j)
				t[l][j] = pInst->transform.m[k++];
		AEGfxSetTransform(t);

		if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_BALL)
		{
			int ttiimmee = (int)timeGetTime();
			ttiimmee %= 5;

			++only4;
			if (only4 > 4)
			{
				AEGfxSetTintColor(1.0f * cosf((float)(i * 2) * PI / (float)ttiimmee), 1.0f, 0.0f, 1.0f);
			}
			else
			{
				AEGfxSetTintColor(1.0f, 0.2f, 0.2f, 1.0f);
			}
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
		}
		else if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_PILLAR)
		{
			AEGfxSetTintColor(1.0f, 1.0f, 1.0f, 1.0f);
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
		}
		else if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_WALL)
		{
			AEGfxSetTintColor(1.0f, 1.0f, 1.0f, 1.0f);
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_LINES_STRIP);
		}
	}
	
	char strBuffer[100];
	memset(strBuffer, 0, 100*sizeof(char));
	sprintf_s(strBuffer, "FPS:  %.6f", 1.0f / AEFrameRateControllerGetFrameTime());
	
	
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	AEGfxTextureSet(NULL, 0, 0);	
	AEGfxSetTransparency(1.0f);
	
	//AEGfxPrint(fontId, strBuffer, -0.95f, -0.95f, 2.0f, 1.f, 0.f, 1.f);
	AEGfxPrint(fontId, strBuffer, (270.0f) / (float)(AEGetWindowWidth() / 2), (350.0f) / (float)(AEGetWindowHeight() / 2), 1.0f, 1.f, 0.f, 0.f);
}

/*!*****************************************************************************
\brief
	Set the flag of all game object instance to false and deallocate all 
	the memory allocated for sBallData, sWallData and sPillarData
*******************************************************************************/
void GameStateCageFree(void)
{
	// kill all object in the list
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
		gameObjInstDestroy(sGameObjInstList + i);

	delete[] sBallData;
	sBallData = NULL;
	
	delete[] sWallData;
	sWallData = NULL;

	delete[] sPillarData;
	sPillarData = NULL;
}

/*!*****************************************************************************
\brief
	Releases all the memory allocated on the heap
*******************************************************************************/
void GameStateCageUnload(void)
{
	// free all CREATED mesh
	for (u32 i = 0; i < sGameObjNum; i++)
		AEGfxMeshFree(sGameObjList[i].pMesh);

	free(sGameObjInstList);
	free(sGameObjList);
}

/*!*****************************************************************************
\brief
	Create and instance of game object of the specified type
\param [in] type:
	Type of game object to be created
\param [in] scale:
	Size of the game object to be created
\param [in] pPos:
	Initial position of game object to be created
\param [in] pVel:
	Initial velocity of game object to be created
\param [in] dir:
	Initial direction of how much the game object will be rotated when it is
	created
*******************************************************************************/
GameObjInst* gameObjInstCreate(TYPE_OBJECT type,
							   float scale, 
							   CSD1130::Vec2* pPos,
							   CSD1130::Vec2* pVel,
							   float dir)
{
	CSD1130::Vec2 zero;

	AE_ASSERT_PARM(type < TYPE_OBJECT(sGameObjNum));
	
	// loop through the object instance list to find a non-used object instance
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// check if current instance is not used
		if ((pInst->flag & FLAG_ACTIVE) == 0)
		{
			// it is not used => use it to create the new instance
			pInst->pObject			 = sGameObjList + (int)type;
			pInst->flag				 = FLAG_ACTIVE | FLAG_VISIBLE;
			pInst->scale			 = scale;
			pInst->posCurr			 = pPos ? *pPos : zero;
			pInst->velCurr			 = pVel ? *pVel : zero;
			pInst->dirCurr			 = dir;
			pInst->pUserData		 = 0;
			
			// return the newly created instance
			return pInst;
		}
	}

	return 0;
}

/*!*****************************************************************************
\brief
	Set the active state of this game object instance to false
\param [out] pInst:
	Game object instance to set it's active flag to false
*******************************************************************************/
void gameObjInstDestroy(GameObjInst* pInst)
{
	// if instance is destroyed before, just return
	if (pInst->flag == 0)
		return;

	// zero out the flag
	pInst->flag = 0;
}