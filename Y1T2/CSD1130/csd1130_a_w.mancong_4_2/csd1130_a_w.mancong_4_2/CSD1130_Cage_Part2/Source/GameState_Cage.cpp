/*!*****************************************************************************
\file		GameState_Cage.cpp
\author 	Wong Man Cong
\par    	DP email: w.mancong\@digipen.edu
\date   	26-03-2022
\brief
This file contains functions definition for Load, Init, Update, Draw, Free and Unload
resources that simulates collision between a line segment and a ball

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

const float			PI_OVER_180				= PI/180.0f;


enum class TYPE_OBJECT
{
	TYPE_OBJECT_BALL,	//0
	TYPE_OBJECT_WALL,	//1

	TYPE_OBJECT_NUM
};

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/
struct GameObj
{
	TYPE_OBJECT		type;		// object type
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
	Create an instance of game object based on it's type
\param [in] type:
	Type of game object to be created
\param [in] scale:
	Size of the game object in game
\param [in] pPos:
	Initial position of the game object
\param [in] pVel:
	Initial velocity of the game object
\param [in] dir:
	Where the game object will be facing initially
\return
	A pointer to the game object created upon success, else return nullptr
*******************************************************************************/
GameObjInst*		gameObjInstCreate (TYPE_OBJECT type,
										float scale, 
										CSD1130::Vector2D* pPos, 
										CSD1130::Vector2D* pVel, 
										float dir);

/*!*****************************************************************************
\brief
	Set the active state of pInst to false
\param [in] pInst:
	Game Object Instance to set it's active state to false
*******************************************************************************/
void				gameObjInstDestroy(GameObjInst* pInst);

static Circle			*sBallData = 0;
static LineSegment		*sWallData = 0;



/*!*****************************************************************************
\brief
	Load function for GameState Cage
*******************************************************************************/
void GameStateCageLoad(void)
{
	sGameObjList = (GameObj *)calloc(GAME_OBJ_NUM_MAX, sizeof(GameObj));
	sGameObjInstList = (GameObjInst *)calloc(GAME_OBJ_INST_NUM_MAX, sizeof(GameObjInst));
	sGameObjNum = 0;

	GameObj* pObj;

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
		int Parts = 36;
		for (float i = 0; i < Parts; ++i)
		{
			AEGfxTriAdd(
				0.0f, 0.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf(i * 2 * PI / Parts) * 1.0f, sinf(i * 2 * PI / Parts) * 1.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf((i + 1) * 2 * PI / Parts) * 1.0f, sinf((i + 1) * 2 * PI / Parts) * 1.0f, 0xFFFFFF00, 0.0f, 0.0f);
		}

		pObj->pMesh = AEGfxMeshEnd();
	}

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
	AEGfxSetBackgroundColor(0.2f, 0.2f, 0.2f);	
}

/*!*****************************************************************************
\brief
	Init function for GameState Cage
*******************************************************************************/
void GameStateCageInit(void)
{
	GameObjInst *pInst;
	std::string str;
	
	std::ifstream inFile("..\\Bin\\Resources\\LevelData.txt");
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
			// create ball instance
			CSD1130::Vector2D vel = CSD1130::Vector2D(cos(dir * PI_OVER_180) * speed, sin(dir * PI_OVER_180) * speed);
			pInst = gameObjInstCreate(TYPE_OBJECT::TYPE_OBJECT_BALL, sBallData[i].m_radius,
										&sBallData[i].m_center, &vel, 0.0f);
			AE_ASSERT(pInst);
			pInst->speed = speed;
			pInst->pUserData = &sBallData[i];
		}

		// read wall data
		unsigned int wallNum = 0;
		CSD1130::Vector2D pos;

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
	Update function for GameState Cage
	Check collision between ball and line segment, and if they collide reflect
	the ball velocity
*******************************************************************************/
void GameStateCageUpdate(void)
{
	static bool full_screen_me;
	if (AEInputCheckTriggered(AEVK_F))
	{
		full_screen_me = !full_screen_me;
		AEToogleFullScreen(full_screen_me);
	}


	CSD1130::Vector2D	interPt;
	CSD1130::Vector2D   normalAtCollision;
	float				interTime = 0.0f;

	//Update object instances positions
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst *pBallInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pBallInst->flag & FLAG_ACTIVE) ||
			pBallInst->pObject->type != TYPE_OBJECT::TYPE_OBJECT_BALL)
			continue;

		CSD1130::Vector2D posNext;
		posNext.x = pBallInst->posCurr.x + pBallInst->velCurr.x * g_dt;
		posNext.y = pBallInst->posCurr.y + pBallInst->velCurr.y * g_dt;

		// Update the latest ball data with the lastest ball's position
		Circle &ballData = *((Circle*)pBallInst->pUserData);
		ballData.m_center.x = pBallInst->posCurr.x;
		ballData.m_center.y = pBallInst->posCurr.y;

		// Check collision with walls
		for(unsigned int j = 0; j < GAME_OBJ_INST_NUM_MAX; ++j)
		{
			GameObjInst *pWallInst = sGameObjInstList + j;

			if (0 == (pWallInst->flag & FLAG_ACTIVE) ||
				pWallInst->pObject->type != TYPE_OBJECT::TYPE_OBJECT_WALL)
				continue;

			LineSegment &lineSegData = *((LineSegment*)pWallInst->pUserData);

			if(CollisionIntersection_CircleLineSegment(ballData, 
													   posNext, 
													   lineSegData, 
													   interPt, 
													   normalAtCollision,
													   interTime))
			{
				CSD1130::Vector2D reflectedVec;

				CollisionResponse_CircleLineSegment(interPt, 
													normalAtCollision,
													posNext, 
													reflectedVec);

				pBallInst->velCurr.x = reflectedVec.x * pBallInst->speed;
				pBallInst->velCurr.y = reflectedVec.y * pBallInst->speed;
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
}

/*!*****************************************************************************
\brief
	Render function for GameState Cage
*******************************************************************************/
void GameStateCageDraw(void)
{
	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxTextureSet(NULL, 0, 0);
	AEGfxSetTransparency(1.0f);	
	//Drawing the object instances
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
			AEGfxSetTintColor(1.0f * cosf((float)(i * 2) * PI / (float)ttiimmee), 1.0f, 0.0f, 1.0f);
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
		}
		else if (pInst->pObject->type == TYPE_OBJECT::TYPE_OBJECT_WALL)
		{
			AEGfxSetTintColor(1.0f, 1.0f, 1.0f, 1.0f);
			AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_LINES_STRIP);
		}
	}

	char strBuffer[100];
	memset(strBuffer, 0, 100 * sizeof(char));
	sprintf_s(strBuffer, "FPS:  %.6f", 1.0 / AEFrameRateControllerGetFrameTime());

	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	AEGfxTextureSet(NULL, 0, 0);
	AEGfxSetTransparency(1.0f);	
	
	AEGfxPrint(fontId, strBuffer, (270.0f) / (float)(AEGetWindowWidth() / 2), (350.0f) / (float)(AEGetWindowHeight() / 2), 1.0f, 1.f, 0.f, 0.f);
}

/*!*****************************************************************************
\brief
	Set the state of all game object instance to false, and deallocate 
	memory allocated to sBallData and sWallData
*******************************************************************************/
void GameStateCageFree(void)
{
	// kill all objects in the list
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
		gameObjInstDestroy(sGameObjInstList + i);

	delete[] sBallData;
	sBallData = nullptr;
	
	delete[] sWallData;
	sWallData = nullptr;
}

/*!*****************************************************************************
\brief
	Deallocating memory allocated onto the heap
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
	Create an instance of game object based on it's type
\param [in] type:
	Type of game object to be created
\param [in] scale:
	Size of the game object in game
\param [in] pPos:
	Initial position of the game object
\param [in] pVel:
	Initial velocity of the game object
\param [in] dir:
	Where the game object will be facing initially
\return
	A pointer to the game object created upon success, else return nullptr
*******************************************************************************/
GameObjInst* gameObjInstCreate(TYPE_OBJECT type,
							   float scale, 
							   CSD1130::Vector2D* pPos, 
							   CSD1130::Vector2D* pVel, 
							   float dir)
{
	CSD1130::Vector2D zero;

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

	return nullptr;
}

/*!*****************************************************************************
\brief
	Set the active state of pInst to false
\param [in] pInst:
	Game Object Instance to set it's active state to false
*******************************************************************************/
void gameObjInstDestroy(GameObjInst* pInst)
{
	// if instance is destroyed before, just return
	if (pInst->flag == 0)
		return;

	// zero out the flag
	pInst->flag = 0;
}