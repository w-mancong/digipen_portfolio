/******************************************************************************/
/*!
\file		GameState_Platformer.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	07-03-2022
\brief		This file contains implementation for a simple platformer game

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"
#include <iostream>
#include <fstream>
#include <string>

/******************************************************************************/
/*!
	Defines
*/
/******************************************************************************/
const unsigned int	GAME_OBJ_NUM_MAX		= 32;	//The total number of different objects (Shapes)
const unsigned int	GAME_OBJ_INST_NUM_MAX	= 2048;	//The total number of different game object instances

//Gameplay related variables and values
const float			GRAVITY					= -20.0f;
const float			JUMP_VELOCITY			= 11.0f;
const float			MOVE_VELOCITY_HERO		= 4.0f;
const float			MOVE_VELOCITY_ENEMY		= 7.5f;
const double		ENEMY_IDLE_TIME			= 2.0;
const int			HERO_LIVES				= 3;

// Particles
const int			PARTICLE_SPAWNS			= 2;
const float			PARTICLE_VELOCITY_X		= 2.5f;
const float			PARTICLE_VELOCITY_Y		= 3.25f;
const float			PARTICLE_SPAWN_SIZE		= 0.3f;
const float			PARTICLE_DECREASE		= 0.35f;

//Flags
const unsigned int	FLAG_ACTIVE				= 0x00000001;
const unsigned int	FLAG_VISIBLE			= 0x00000002;
const unsigned int	FLAG_NON_COLLIDABLE		= 0x00000004;

//Collision flags
const unsigned int	COLLISION_LEFT			= 0x00000001;	//0001
const unsigned int	COLLISION_RIGHT			= 0x00000002;	//0010
const unsigned int	COLLISION_TOP			= 0x00000004;	//0100
const unsigned int	COLLISION_BOTTOM		= 0x00000008;	//1000


enum TYPE_OBJECT
{
	TYPE_OBJECT_EMPTY,			//0
	TYPE_OBJECT_COLLISION,		//1
	TYPE_OBJECT_HERO,			//2
	TYPE_OBJECT_ENEMY1,			//3
	TYPE_OBJECT_COIN,			//4
	TYPE_OBJECT_PARTICLE1,		//5
	TYPE_OBJECT_PARTICLE2		//6
};

//State machine states
enum STATE
{
	STATE_NONE,
	STATE_GOING_LEFT,
	STATE_GOING_RIGHT
};

//State machine inner states
enum INNER_STATE
{
	INNER_STATE_ON_ENTER,
	INNER_STATE_ON_UPDATE,
	INNER_STATE_ON_EXIT
};

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/
struct GameObj
{
	unsigned int		type;		// object type
	AEGfxVertexList*	pMesh;		// pbject
};


struct GameObjInst
{
	GameObj*		pObject;	// pointer to the 'original'
	unsigned int	flag;		// bit flag or-ed together
	float			scale;
	AEVec2			posCurr;	// object current position
	AEVec2			velCurr;	// object current velocity
	float			dirCurr;	// object current direction

	AEMtx33			transform;	// object drawing matrix
	
	AABB			boundingBox;// object bouding box that encapsulates the object

	//Used to hold the current 
	int				gridCollisionFlag;

	// pointer to custom data specific for each object type
	void*			pUserData;

	//State of the object instance
	enum			STATE state;
	enum			INNER_STATE innerState;

	//General purpose counter (This variable will be used for the enemy state machine)
	double			counter;
};


/******************************************************************************/
/*!
	File globals
*/
/******************************************************************************/
static int				HeroLives;
static int				Hero_Initial_X;
static int				Hero_Initial_Y;
static int				TotalCoins;
static bool				arielJump;
//We need a pointer to the hero's instance for input purposes
static GameObjInst*		pHero;

// list of original objects
static GameObj*			sGameObjList;
static unsigned int		sGameObjNum;

// list of object instances
static GameObjInst*		sGameObjInstList;
static unsigned int		sGameObjInstNum;

//Binary map data
static int**			MapData;
static int**			BinaryCollisionArray;
static int				BINARY_MAP_WIDTH;
static int				BINARY_MAP_HEIGHT;
static GameObjInst*		pBlackInstance;
static GameObjInst*		pWhiteInstance;
static AEMtx33			MapTransform;

// Particles
static float			ParticlePosX;

/*!**************************************************************************************
\brief
	Get the value store inside the position X and Y of MapData array

\param [in] X
	X coordinate of the array

\param [in] Y
	Y coordinate of the array

\return
	Return either 0 or 1 based on the position in the array
****************************************************************************************/
int						GetMapDataValue(int x, int y);

/*!**************************************************************************************
\brief
	Get the value store inside the position X and Y of BinaryCollisionArray

\param [in] X
	X coordinate of the array

\param [in] Y
	Y coordinate of the array

\return
	Return either 0 or 1 based on the position in the array
****************************************************************************************/
int						GetCellValue(int x, int y);

/*!**************************************************************************************
\brief
	Based on the current position and size of the cube, calculate the hotspots
	of the cube and determine if it has any collision using the datas store inside
	BinaryCollisionArray. If any of the hotspots are inside the cell of the
	collision array:
	-> The COLLISION_LEFT, COLLISION_RIGHT, COLLISION_TOP, COLLISION_BOTTOM flags
	will be flagged accordingly

\param [in] PosX
	X position of the cube

\param [in] PosY
	Y position of the cube

\param [in] scaleX
	Size of the cube horizontally

\param [in] scaleY
	Size of the cube vertically

\return
	A flag number determining where if there is collision on the TOP, BOTTOM,
	RIGHT and LEFT edges of the cube
****************************************************************************************/
int						CheckInstanceBinaryMapCollision(float PosX, float PosY, 
														float scaleX, float scaleY);

/*!**************************************************************************************
\brief
	Snaps the Coordinate back to it's original position

\param [in, out] Coordinate
	The coordinate to be snapped back to it's position
****************************************************************************************/
void					SnapToCell(float *Coordinate);

/*!**************************************************************************************
\brief
	Load data from text file and storing it into 2D dynamic array

\param [in] FileName
	Name of the file to have data to be extrated and stored into the
	MapData and BinaryCollisionArray

\return
	False (0) if FileName does not exist else True (1)
****************************************************************************************/
int						ImportMapDataFromFile(char *FileName);

/*!**************************************************************************************
\brief
	Deallocate dynamically allocated memory for MapData & BinaryCollisionArray
****************************************************************************************/
void					FreeMapData(void);

/*!**************************************************************************************
\brief
	Create an instance of game object based on the arguments

\param [in] type
	Type of game object to be created
\param [in] scale
	Size of game object
\param [in] pPos
	Initial position of game object when it's instantiated
\param [in] pVel
	Initial velocity of game object when it's instantiated
\param [in] dir
	Initial direction of game object when it's instantiated
\param [in] startState
	Initial state of game object when it's instantiated

\return
	A pointer to a GameObjInst if a game object is successfully created. Else, it will
	return a nullptr.
****************************************************************************************/
static GameObjInst*		gameObjInstCreate (unsigned int type, float scale, 
											AEVec2* pPos, AEVec2* pVel, 
											float dir, enum STATE startState);

/*!**************************************************************************************
\brief
	Set the flag of pInst to 0 (Deactivating the game object)

\param [in] pInst
	Pointer to the game object to be deactivated
****************************************************************************************/
static void				gameObjInstDestroy(GameObjInst* pInst);

/*!**************************************************************************************
\brief
	Controls the behaviour of enemy

\param [in] pInst
	Pointer to an instance of Game Object of type Enemy
****************************************************************************************/
void					EnemyStateMachine(GameObjInst *pInst);

/*!**************************************************************************************
\brief
	Randomize an integer between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
****************************************************************************************/
int						Random(int min, int max);

/*!**************************************************************************************
\brief
	Randomize a float between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
****************************************************************************************/
float					Random(float min, float max);

/*!**************************************************************************************
\brief
	Load function of my platformer game
	- Allocates memory on the heap
	- Creating meshes for individual game objects
	- Load in MapData and BinaryCollisionArray from text file
	- Initialize MapTransform matrix
****************************************************************************************/
void GameStatePlatformLoad(void)
{
	sGameObjList		= (GameObj *)calloc(GAME_OBJ_NUM_MAX, sizeof(GameObj));
	sGameObjInstList	= (GameObjInst *)calloc(GAME_OBJ_INST_NUM_MAX, sizeof(GameObjInst));
	sGameObjNum			= 0;

	GameObj* pObj;

	//Creating the black object
	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_EMPTY;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFF000000, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFF000000, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF000000, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFF000000, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create empty object!!");
	}
	
	//Creating the white object
	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_COLLISION;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create collision object!!");
	}

	//Creating the hero object
	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_HERO;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFF0000FF, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFF0000FF, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF0000FF, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFF0000FF, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create hero object!!");
	}
		//Creating the enemey1 object
	pObj				= sGameObjList + sGameObjNum++;
	if(pObj)
	{
		pObj->type		= TYPE_OBJECT_ENEMY1;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFFFF0000, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFFFF0000, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFFFF0000, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create enemy object!!");
	}

	//Creating the Coin object
	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_COIN;

		AEGfxMeshStart();
		//Creating the circle shape
		int Parts = 12;
		for (float i = 0; i < Parts; ++i)
		{
			AEGfxTriAdd(
				0.0f, 0.0f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf(i * 2 * PI / Parts) * 0.5f, sinf(i * 2 * PI / Parts) * 0.5f, 0xFFFFFF00, 0.0f, 0.0f,
				cosf((i + 1) * 2 * PI / Parts) * 0.5f, sinf((i + 1) * 2 * PI / Parts) * 0.5f, 0xFFFFFF00, 0.0f, 0.0f);
		}

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create coin object!!");
	}

	// Creating particle object
	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_PARTICLE1;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF416687, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF416687, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFF416687, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFF416687, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF416687, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFF416687, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create first particle object!!");
	}

	pObj				= sGameObjList + sGameObjNum++;
	if (pObj)
	{
		pObj->type		= TYPE_OBJECT_PARTICLE2;

		AEGfxMeshStart();
		AEGfxTriAdd(
			-0.5f, -0.5f, 0xFF6EAD8A, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF6EAD8A, 0.0f, 0.0f,
			-0.5f,  0.5f, 0xFF6EAD8A, 0.0f, 0.0f);

		AEGfxTriAdd(
			-0.5f,  0.5f, 0xFF6EAD8A, 0.0f, 0.0f,
			 0.5f, -0.5f, 0xFF6EAD8A, 0.0f, 0.0f,
			 0.5f,  0.5f, 0xFF6EAD8A, 0.0f, 0.0f);

		pObj->pMesh = AEGfxMeshEnd();
		AE_ASSERT_MESG(pObj->pMesh, "fail to create second particle object!!");
	}

	//Setting intital binary map values
	MapData = 0;
	BinaryCollisionArray = 0;
	BINARY_MAP_WIDTH = 0;
	BINARY_MAP_HEIGHT = 0;

	char fileName[128]{ '\0' };
	//Importing Data
	switch (gGameStateCurr)
	{
		case GS_PLATFORM_1:
		{
			strcpy_s(fileName, "../Resources/Levels/Exported.txt");
			break;
		}
		case GS_PLATFORM_2:
		{
			strcpy_s(fileName, "../Resources/Levels/Exported2.txt");
			break;
		}
	}
	if(!ImportMapDataFromFile(fileName))
		gGameStateNext = GS_QUIT;

	AEMtx33 scale, trans;
	AEMtx33Trans(&trans, static_cast<float>(-BINARY_MAP_WIDTH >> 1), static_cast<float>(-BINARY_MAP_HEIGHT >> 1));
	AEMtx33Scale(&scale, static_cast<float>(AEGetWindowWidth()) * 0.05f, static_cast<float>(AEGetWindowHeight()) * 0.05f);
	AEMtx33Concat(&MapTransform, &scale, &trans);
}

/*!**************************************************************************************
\brief
	Init function of my platformer game
	- Initializes all the necessary data to it's default value
****************************************************************************************/
void GameStatePlatformInit(void)
{
	system("cls");
	std::cout << "Welcome to my mini platformer game!" << std::endl;
	std::cout << "Author: Wong Man Cong" << std::endl;
	std::cout << "Email: w.mancong@digipen.edu" << std::endl;

	pHero = 0;
	g_dt = 0.0f;
	pBlackInstance = 0;
	pWhiteInstance = 0;
	TotalCoins = 0;
	arielJump = false;
	ParticlePosX = -1.0f;

	pBlackInstance = gameObjInstCreate(TYPE_OBJECT_EMPTY, 1.0f, 0, 0, 0.0f, STATE_NONE);
	pBlackInstance->flag ^= FLAG_VISIBLE;
	pBlackInstance->flag |= FLAG_NON_COLLIDABLE;

	pWhiteInstance = gameObjInstCreate(TYPE_OBJECT_COLLISION, 1.0f, 0, 0, 0.0f, STATE_NONE);
	pWhiteInstance->flag ^= FLAG_VISIBLE;
	pWhiteInstance->flag |= FLAG_NON_COLLIDABLE;

	//Setting the inital number of hero lives
	HeroLives = HERO_LIVES;

	AEVec2 Pos;
	for (int x = 0; x < BINARY_MAP_WIDTH; ++x)
	{
		for (int y = 0; y < BINARY_MAP_HEIGHT; ++y)
		{
			TYPE_OBJECT type = static_cast<TYPE_OBJECT>(GetMapDataValue(x, y));
			AEVec2Set(&Pos, x + 0.5f, y + 0.5f);
			switch (type)
			{
				case TYPE_OBJECT_HERO:
				{
					Hero_Initial_X = x;
					Hero_Initial_Y = y;
					pHero = gameObjInstCreate(type, 1.0f, &Pos, 0, 0.0f, STATE_NONE);
					break;
				}
				case TYPE_OBJECT_ENEMY1:
				{
					gameObjInstCreate(type, 1.0f, &Pos, 0, 0.0f, STATE_GOING_LEFT);
					break;
				}
				case TYPE_OBJECT_COIN:
				{
					++TotalCoins;
					gameObjInstCreate(type, 1.0f, &Pos, 0, 0.0f, STATE_NONE);
					break;
				}
				default:
					continue;
			}
		}
	}
}

/*!**************************************************************************************
\brief
	Update function of my platformer game
	- Update player's input
	- Spawns particle
	- Update status of particles for when to despawn it
	- Apply gravity to all necessary game objects
	- Update enemy state machine
	- Update position of game object based on their velocity
	- Update bounding box
	- Update binary collision
	- Update aabb collision
	- Computes transformation matrix of each game object
	- Clamp camera's position
****************************************************************************************/
void GameStatePlatformUpdate(void)
{
	if (AEInputCheckTriggered(AEVK_ESCAPE))
		gGameStateNext = GS_MENU;

	GameObjInst *pInst;

	AEVec2Set(&pHero->velCurr, 0.0f, pHero->velCurr.y);
	if (AEInputCheckCurr(AEVK_LEFT))
	{
		ParticlePosX = 1.0f;
		AEVec2Set(&pHero->velCurr, -MOVE_VELOCITY_HERO, pHero->velCurr.y);
	}
	else if (AEInputCheckCurr(AEVK_RIGHT))
	{
		ParticlePosX = -1.0f;
		AEVec2Set(&pHero->velCurr, MOVE_VELOCITY_HERO, pHero->velCurr.y);
	}

	if (AEInputCheckTriggered(AEVK_SPACE) && (pHero->gridCollisionFlag & COLLISION_BOTTOM))
	{
		arielJump = true;
		AEVec2Set(&pHero->velCurr, pHero->velCurr.x, JUMP_VELOCITY);
	}

	if (arielJump)
	{
		if ((pHero->gridCollisionFlag & COLLISION_LEFT))
		{
			if (AEInputCheckCurr(AEVK_RIGHT) && AEInputCheckTriggered(AEVK_SPACE))
				AEVec2Set(&pHero->velCurr, pHero->velCurr.x, JUMP_VELOCITY);
		}
		else if ((pHero->gridCollisionFlag & COLLISION_RIGHT))
		{
			if (AEInputCheckCurr(AEVK_LEFT) && AEInputCheckTriggered(AEVK_SPACE))
				AEVec2Set(&pHero->velCurr, pHero->velCurr.x, JUMP_VELOCITY);
		}
	}

	// Spawn Particles
	for (int i = 0; i < PARTICLE_SPAWNS; ++i)
	{
		const float rand_x	= Random(-0.2f, 0.2f), rand_y = Random(0.0f, 0.2f);
		const float size	= Random(PARTICLE_SPAWN_SIZE, PARTICLE_SPAWN_SIZE + 0.2f);
		unsigned int type	= Random(0, 2);
		AEVec2 pos{ pHero->posCurr.x + rand_x + pHero->scale * 0.25f * ParticlePosX,  pHero->posCurr.y + rand_y + pHero->scale * 0.5f };
		AEVec2 vel{ PARTICLE_VELOCITY_X * ParticlePosX, PARTICLE_VELOCITY_Y };
		gameObjInstCreate(TYPE_OBJECT_PARTICLE1 + type, size, &pos, &vel, 0.0f, STATE_NONE);
	}

	// Update particles size
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;
		if (!(pInst->flag & FLAG_ACTIVE) || pInst->pObject->type < TYPE_OBJECT_PARTICLE1)
			continue;
		pInst->scale -= PARTICLE_DECREASE * g_dt;
		if (pInst->scale <= 0.0f)
			gameObjInstDestroy(pInst);
	}

	//Update object instances physics and behavior
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		if (pInst->pObject->type == TYPE_OBJECT_COIN || pInst->pObject->type >= TYPE_OBJECT_PARTICLE1)
			continue;
		pInst->velCurr.y += GRAVITY * g_dt;
		if (pInst->pObject->type != TYPE_OBJECT_ENEMY1)
			continue;
		EnemyStateMachine(pInst);
	}

	//Update object instances positions
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		AEVec2 vel;
		AEVec2Scale(&vel, &pInst->velCurr, g_dt);
		AEVec2Add(&pInst->posCurr, &pInst->posCurr, &vel);

		pInst->boundingBox.min.x = pInst->posCurr.x - pInst->scale * 0.5f;
		pInst->boundingBox.max.x = pInst->posCurr.x + pInst->scale * 0.5f;
		pInst->boundingBox.min.y = pInst->posCurr.y - pInst->scale * 0.5f;
		pInst->boundingBox.max.y = pInst->posCurr.y + pInst->scale * 0.5f;
	}

	//Check for grid collision
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;

		// skip non-active object instances
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		if (pInst->pObject->type >= TYPE_OBJECT_PARTICLE1)
			continue;
		pInst->gridCollisionFlag = CheckInstanceBinaryMapCollision(pInst->posCurr.x, pInst->posCurr.y, pInst->scale, pInst->scale);
		if ((pInst->gridCollisionFlag & COLLISION_TOP) || (pInst->gridCollisionFlag & COLLISION_BOTTOM))
		{
			SnapToCell(&pInst->posCurr.y);
			pInst->velCurr.y = 0.0f;
		}
		if ((pInst->gridCollisionFlag & COLLISION_LEFT) || (pInst->gridCollisionFlag & COLLISION_RIGHT))
		{
			SnapToCell(&pInst->posCurr.x);
			pInst->velCurr.x = 0.0f;
		}
	}

	if ((pHero->gridCollisionFlag & COLLISION_BOTTOM))
		arielJump = false;
	
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		pInst = sGameObjInstList + i;
		if (!(pInst->flag & FLAG_ACTIVE) || (pInst->flag & FLAG_NON_COLLIDABLE))
			continue;

		if (!CollisionIntersection_RectRect(pHero->boundingBox, pHero->velCurr, pInst->boundingBox, pInst->velCurr))
			continue;
		switch (pInst->pObject->type)
		{
			case TYPE_OBJECT_ENEMY1:
			{
				--HeroLives;
				if (HeroLives <= 0)
					gGameStateNext = GS_RESTART;
				AEVec2Set(&pHero->posCurr, static_cast<float>(Hero_Initial_X), static_cast<float>(Hero_Initial_Y));
				break;
			}
			case TYPE_OBJECT_COIN:
			{
				--TotalCoins;
				if (TotalCoins <= 0)
					gGameStateNext = GS_MENU;
				gameObjInstDestroy(pInst);
				break;
			}
		}
	}
	
	//Computing the transformation matrices of the game object instances
	for(unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		AEMtx33 scale, rot, trans;
		pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE))
			continue;

		AEMtx33Scale(&scale, pInst->scale, pInst->scale);
		AEMtx33Rot(&rot, pInst->dirCurr);
		AEMtx33Trans(&trans, pInst->posCurr.x, pInst->posCurr.y);
		AEMtx33Concat(&pInst->transform, &rot, &scale);
		AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);
	}
	
	AEVec2 cam_pos;
	AEVec2 const min_pos{ 10.0f, 10.0f }, max_pos{ static_cast<float>(BINARY_MAP_WIDTH) - 10.0f, static_cast<float>(BINARY_MAP_HEIGHT) - 10.0f };
	cam_pos.x = AEClamp(pHero->posCurr.x, min_pos.x, max_pos.x);
	cam_pos.y = AEClamp(pHero->posCurr.y, min_pos.y, max_pos.y);
	AEMtx33MultVec(&cam_pos, &MapTransform, &cam_pos);
	AEGfxSetCamPosition(cam_pos.x, cam_pos.y);
}

/*!**************************************************************************************
\brief
	Draw function for my platformer game
****************************************************************************************/
void GameStatePlatformDraw(void)
{
	AEMtx33 cellTranslation, cellFinalTransformation;

	AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	for (int x = 0; x < BINARY_MAP_WIDTH; ++x)
	{
		for (int y = 0; y < BINARY_MAP_HEIGHT; ++y)
		{
			AEMtx33Trans(&cellTranslation, static_cast<float>(x + 0.5f), static_cast<float>(y + 0.5f));
			AEMtx33Concat(&cellFinalTransformation, &MapTransform, &cellTranslation);
			AEGfxSetTransform(cellFinalTransformation.m);
			switch (GetCellValue(x, y))
			{
				case TYPE_OBJECT_EMPTY:
				{
					AEGfxMeshDraw(pBlackInstance->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
					break;
				}
				case TYPE_OBJECT_COLLISION:
				{
					AEGfxMeshDraw(pWhiteInstance->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
					break;
				}
				default:
					break;
			}
		}
	}

	for (int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// skip non-active object
		if (0 == (pInst->flag & FLAG_ACTIVE) || 0 == (pInst->flag & FLAG_VISIBLE))
			continue;
		
		//Don't forget to concatenate the MapTransform matrix with the transformation of each game object instance
		//AEMtx33Trans(&cellTranslation, static_cast<float>(pInst->posCurr.x), static_cast<float>(pInst->posCurr.y));
		AEMtx33Concat(&pInst->transform, &MapTransform, &pInst->transform);
		AEGfxSetTransform(pInst->transform.m);
		AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
	}

	char strBuffer[100];
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	sprintf_s(strBuffer, "Lives:  %d", HeroLives);
	f32 TextWidth, TextHeight;
	AEGfxGetPrintSize(fontID, strBuffer, 1.0f, TextWidth, TextHeight);
	AEGfxPrint(fontID, strBuffer, 0.85f - TextWidth, 0.96f - TextHeight, 1.0f, 0.f, 0.f, 1.f);

	sprintf_s(strBuffer, "Coins Left:  %d", TotalCoins);
	AEGfxGetPrintSize(fontID, strBuffer, 1.0f, TextWidth, TextHeight);
	AEGfxPrint(fontID, strBuffer, -0.49f - TextWidth, 0.96f - TextHeight, 1.0f, 0.f, 0.f, 1.f);
}

/*!**************************************************************************************
\brief
	Reset the active flag of all GameObjInst
****************************************************************************************/
void GameStatePlatformFree(void)
{
	// kill all object in the list
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
		gameObjInstDestroy(sGameObjInstList + i);
}

/*!**************************************************************************************
\brief
	Deallocate any memory from the heap
****************************************************************************************/
void GameStatePlatformUnload(void)
{
	// free all CREATED mesh
	for (u32 i = 0; i < sGameObjNum; i++)
		AEGfxMeshFree(sGameObjList[i].pMesh);

	free(sGameObjInstList);
	free(sGameObjList);
	/****************
	Free the map data
	****************/
	FreeMapData();
}

/*!**************************************************************************************
\brief
	Create an instance of game object based on the arguments

\param [in] type
	Type of game object to be created
\param [in] scale
	Size of game object
\param [in] pPos
	Initial position of game object when it's instantiated
\param [in] pVel
	Initial velocity of game object when it's instantiated
\param [in] dir
	Initial direction of game object when it's instantiated
\param [in] startState
	Initial state of game object when it's instantiated

\return
	A pointer to a GameObjInst if a game object is successfully created. Else, it will
	return a nullptr.
****************************************************************************************/
GameObjInst* gameObjInstCreate(unsigned int type, float scale, 
							   AEVec2* pPos, AEVec2* pVel, 
							   float dir, enum STATE startState)
{
	AEVec2 zero;
	AEVec2Zero(&zero);

	AE_ASSERT_PARM(type < sGameObjNum);
	
	// loop through the object instance list to find a non-used object instance
	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// check if current instance is not used
		if (pInst->flag == 0)
		{
			// it is not used => use it to create the new instance
			pInst->pObject			 = sGameObjList + type;
			pInst->flag				 = FLAG_ACTIVE | FLAG_VISIBLE;
			pInst->scale			 = scale;
			pInst->posCurr			 = pPos ? *pPos : zero;
			pInst->velCurr			 = pVel ? *pVel : zero;
			pInst->dirCurr			 = dir;
			pInst->pUserData		 = 0;
			pInst->gridCollisionFlag = 0;
			pInst->state			 = startState;
			pInst->innerState		 = INNER_STATE_ON_ENTER;
			pInst->counter			 = 0;
			
			// return the newly created instance
			return pInst;
		}
	}

	return 0;
}

/*!**************************************************************************************
\brief
	Set the flag of pInst to 0 (Deactivating the game object)

\param [in] pInst
	Pointer to the game object to be deactivated
****************************************************************************************/
void gameObjInstDestroy(GameObjInst* pInst)
{
	// if instance is destroyed before, just return
	if (pInst->flag == 0)
		return;

	// zero out the flag
	pInst->flag = 0;
}

/*!**************************************************************************************
\brief
	Get the value store inside the position X and Y of MapData array

\param [in] X
	X coordinate of the array

\param [in] Y
	Y coordinate of the array

\return
	Return either 0 or 1 based on the position in the array
****************************************************************************************/
int	GetMapDataValue(int x, int y)
{
	if (0 > x || BINARY_MAP_WIDTH <= x || 0 > y || BINARY_MAP_HEIGHT <= y)
		return 0;
	return MapData[y][x];
}

/*!**************************************************************************************
\brief
	Get the value store inside the position X and Y of BinaryCollisionArray

\param [in] X
	X coordinate of the array

\param [in] Y
	Y coordinate of the array

\return
	Return either 0 or 1 based on the position in the array
****************************************************************************************/
int GetCellValue(int x, int y)
{
	if (0 > x || BINARY_MAP_WIDTH <= x || 0 > y || BINARY_MAP_HEIGHT <= y)
		return 0;
	return BinaryCollisionArray[y][x];
}

/*!**************************************************************************************
\brief
	Based on the current position and size of the cube, calculate the hotspots
	of the cube and determine if it has any collision using the datas store inside
	BinaryCollisionArray. If any of the hotspots are inside the cell of the
	collision array:
	-> The COLLISION_LEFT, COLLISION_RIGHT, COLLISION_TOP, COLLISION_BOTTOM flags
	will be flagged accordingly

\param [in] PosX
	X position of the cube

\param [in] PosY
	Y position of the cube

\param [in] scaleX
	Size of the cube horizontally

\param [in] scaleY
	Size of the cube vertically

\return
	A flag number determining where if there is collision on the TOP, BOTTOM,
	RIGHT and LEFT edges of the cube
****************************************************************************************/
int CheckInstanceBinaryMapCollision(float PosX, float PosY, float scaleX, float scaleY)
{
	int flag = 0;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

	// hotspot 1 (right-up)
	x1 = static_cast<int>(PosX + scaleX * 0.5f);
	y1 = static_cast<int>(PosY + scaleY * 0.25f);

	// hotspot 2 (right-down)
	x2 = static_cast<int>(PosX + scaleX * 0.5f);
	y2 = static_cast<int>(PosY - scaleY * 0.25f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag |= COLLISION_RIGHT;

	// hotspot 3 (left-up)
	x1 = static_cast<int>(PosX - scaleX * 0.5f);
	y1 = static_cast<int>(PosY + scaleY * 0.25f);

	// hotspot 4 (left-down)
	x2 = static_cast<int>(PosX - scaleX * 0.5f);
	y2 = static_cast<int>(PosY - scaleY * 0.25f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag |= COLLISION_LEFT;

	// hotspot 5 (up-right)
	x1 = static_cast<int>(PosX + scaleX * 0.25f);
	y1 = static_cast<int>(PosY + scaleY * 0.5f);

	// hotspot 6 (up-leftt)
	x2 = static_cast<int>(PosX - scaleX * 0.25f);
	y2 = static_cast<int>(PosY + scaleY * 0.5f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag |= COLLISION_TOP;

	// hotspot 7 (down-right)
	x1 = static_cast<int>(PosX + scaleX * 0.25f);
	y1 = static_cast<int>(PosY - scaleY * 0.5f);

	// hotspot 8 (down-left)
	x2 = static_cast<int>(PosX - scaleX * 0.25f);
	y2 = static_cast<int>(PosY - scaleY * 0.5f);
	if (GetCellValue(x1, y1) || GetCellValue(x2, y2))
		flag |= COLLISION_BOTTOM;

	return flag;
}

/*!**************************************************************************************
\brief
	Snaps the Coordinate back to it's original position

\param [in, out] Coordinate
	The coordinate to be snapped back to it's position
****************************************************************************************/
void SnapToCell(float *Coordinate)
{
	*Coordinate = static_cast<float>(static_cast<int>(*Coordinate)) + 0.5f;
}

/*!**************************************************************************************
\brief
	Load data from text file and storing it into 2D dynamic array

\param [in] FileName
	Name of the file to have data to be extrated and stored into the
	MapData and BinaryCollisionArray

\return
	False (0) if FileName does not exist else True (1)
****************************************************************************************/
int ImportMapDataFromFile(char *FileName)
{
	std::ifstream ifs(FileName);
	std::string buffer;

	if (!ifs.is_open())
		return false;

	int count = 0;
	const int DATA = 2;	// just tryna loop to get my width and height value
	int temp[DATA] = {};
	while (count < DATA)
	{
		ifs >> buffer;
		if (std::isdigit(buffer[0]))
			temp[count++] = std::stoi(buffer);
	}

	BINARY_MAP_WIDTH = temp[0];	// col
	BINARY_MAP_HEIGHT = temp[1];	// row

	// Allocating memory for my 2d array
	MapData = new int* [(size_t)BINARY_MAP_HEIGHT * (size_t)BINARY_MAP_WIDTH];
	int* ptr = (int*)(MapData + BINARY_MAP_HEIGHT);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
		*(MapData + i) = (ptr + (size_t)BINARY_MAP_WIDTH * i);

	BinaryCollisionArray = new int* [(size_t)BINARY_MAP_HEIGHT * (size_t)BINARY_MAP_WIDTH];
	ptr = (int*)(BinaryCollisionArray + BINARY_MAP_HEIGHT);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
		*(BinaryCollisionArray + i) = (ptr + (size_t)BINARY_MAP_WIDTH * i);

	for (size_t i = 0; i < BINARY_MAP_HEIGHT; ++i)
	{
		for (size_t j = 0; j < BINARY_MAP_WIDTH; ++j)
		{
			ifs >> buffer;
			int num = std::stoi(buffer);

			*(*(MapData + i) + j) = num;
			*(*(BinaryCollisionArray + i) + j) = num == 1 ? 1 : 0;
		}
	}

	ifs.close();

	return true;
}

/*!**************************************************************************************
\brief
	Deallocate dynamically allocated memory for MapData & BinaryCollisionArray
****************************************************************************************/
void FreeMapData(void)
{
	delete[] MapData;
	delete[] BinaryCollisionArray;
}

/*!**************************************************************************************
\brief
	Controls the behaviour of enemy

\param [in] pInst
	Pointer to an instance of Game Object of type Enemy
****************************************************************************************/
void EnemyStateMachine(GameObjInst *pInst)
{
	float x{ 0.0f }, y{ pInst->posCurr.y - 1.0f }, movement_speed{ 0.0f };
	unsigned int col_flag{ 0 };
	STATE next_state{ STATE_NONE };
	switch (pInst->state)
	{
		case STATE_GOING_LEFT:
		{
			movement_speed	= -MOVE_VELOCITY_ENEMY;
			x				= pInst->posCurr.x - 0.5f;
			col_flag		= COLLISION_LEFT;
			next_state		= STATE_GOING_RIGHT;
			break;
		}
		case STATE_GOING_RIGHT:
		{
			movement_speed	= MOVE_VELOCITY_ENEMY;
			x				= pInst->posCurr.x + 0.5f;
			col_flag		= COLLISION_RIGHT;
			next_state		= STATE_GOING_LEFT;
			break;
		}
	}

	switch (pInst->innerState)
	{
		case INNER_STATE_ON_ENTER:
		{
			AEVec2Set(&pInst->velCurr, movement_speed, pInst->velCurr.y);
			pInst->innerState = INNER_STATE_ON_UPDATE;
			break;
		}
		case INNER_STATE_ON_UPDATE:
		{
			if ((pInst->gridCollisionFlag & col_flag) || !GetCellValue(static_cast<int>(x), static_cast<int>(y)))
			{
				pInst->counter = 0.0;
				AEVec2Set(&pInst->velCurr, 0.0f, pInst->velCurr.y);
				pInst->innerState = INNER_STATE_ON_EXIT;
			}
			break;
		}
		case INNER_STATE_ON_EXIT:
		{
			pInst->counter += g_dt;
			if (pInst->counter >= ENEMY_IDLE_TIME)
			{
				pInst->state = next_state;
				pInst->innerState = INNER_STATE_ON_ENTER;
			}
			break;
		}
	}
}

/*!******************************************************************************
\brief
	Randomize an integer between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
*******************************************************************************/
int Random(int min, int max)
{
	return rand() % max + min;
}

/*!******************************************************************************
\brief
	Randomize a float between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
*******************************************************************************/
float Random(float min, float max)
{
	return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}