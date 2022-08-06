/******************************************************************************/
/*!
\file		GameState_Asteroids.cpp
\author 	Wong Man Cong
\par    	email: w.mancong\@digipen.edu
\date   	11-02-2022
\brief		Basic asteroid game 

Copyright (C) 2022 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"
#include <iostream>

/******************************************************************************/
/*!
	Defines
*/
/******************************************************************************/
const unsigned int	GAME_OBJ_NUM_MAX				= 32;			// The total number of different objects (Shapes)
const unsigned int	GAME_OBJ_INST_NUM_MAX			= 4096;			// The total number of different game object instances
const unsigned int	PARTICLES_NUM_MAX				= 1024;			// The total number of particles that can be spawned

const unsigned int	SHIP_INITIAL_NUM				= 3;			// initial number of ship lives
const float			SHIP_SIZE						= 16.0f;		// ship size
const float			SHIP_ACCEL_FORWARD				= 100.0f;		// ship forward acceleration (in m/s^2)
const float			SHIP_ACCEL_BACKWARD				= 100.0f;		// ship backward acceleration (in m/s^2)
const float			SHIP_ROT_SPEED					= (2.0f * PI);	// ship rotation speed (degree/second)
const float			VELOCITY_CAP_PERCENT			= 0.99f;		// percentage to be multiplied to "slow down" ship

const float			BULLET_SPEED					= 750.0f;		// bullet speed (m/s)
const float			BULLET_SIZE						= 10.0f;		// size of bullet

const unsigned long	SCORE_PER_DESTROY				= 100;			// the score when destroying an asteroid
const unsigned long SCORE_TO_WIN					= 5000;			// score to win the game

const float			ASTEROID_MIN_VELOCITY			= 50.0f;		// Minimum velocity for asteroid
const float			ASTEROID_MAX_VELOCITY			= 200.0f;		// Maximum velocity for asteroid
const int			ASTEROID_MIN_SIZE				= 25.0f;		// Minimum size for asteroid
const int			ASTEROID_MAX_SIZE				= 75.0f;		// Maximum size for asteroid

const float			HEART_SIZE						= 25.0f;		// Size of heart texture

const float			PARTICLE_SCALE_INCREASE			= 0.5f;			// Size of particle increasing
const float			SIZE_MOVING_PARTICLE_MIN		= 5.0f;			// Minimum Size of moving ship particle
const float			SIZE_MOVING_PARTICLE_MAX		= 10.0f;		// Maximum Size of moving ship particle
const float			SPEED_MOVING_PARTICLE			= 50.0f;		// Speed of moving particle
const float			PARTICLE_TIME_SCALE				= 5.0f;			// "Duration" of particles
const double		ANGLE_OF_PARTICLE_SPAWN			= 45.0f;		// Angle of particle spawning behind ship

// -----------------------------------------------------------------------------
enum TYPE
{
	// list of game object types
	TYPE_SHIP = 0, 
	TYPE_BULLET,
	TYPE_ASTEROID,
	TYPE_PARTICLE_MOVING,
	TYPE_PARTICLE_EXPLODING,
	TYPE_HEART,

	TYPE_NUM
};

// -----------------------------------------------------------------------------
// object flag definition

const unsigned long FLAG_ACTIVE				= 0x00000001;

/******************************************************************************/
/*!
	Struct/Class Definitions
*/
/******************************************************************************/

struct Color
{
	float r, g, b, a;
};

//Game object structure
struct GameObj
{
	unsigned long		type;		// object type
	AEGfxVertexList *	pMesh;		// This will hold the triangles which will form the shape of the object
};

// ---------------------------------------------------------------------------

//Game object instance structure
struct GameObjInst
{
	GameObj *			pObject;	// pointer to the 'original' shape
	unsigned long		flag;		// bit flag or-ed together
	float				scale;		// scaling value of the object instance
	AEVec2				posCurr;	// object current position
	AEVec2				velCurr;	// object current velocity
	float				dirCurr;	// object current direction
	AABB				boundingBox;// object bouding box that encapsulates the object
	AEMtx33				transform;	// object transformation matrix: Each frame, 
									// calculate the object instance's transformation matrix and save it here
	Color				color;		// tint color of the mesh
	AEGfxTexture*		texture;	// texture for the mesh

	//void				(*pfUpdate)(void);
	//void				(*pfDraw)(void);
};

struct Matrix2x2
{
	float m[2][2];	/*	mat2D = {	a, b
									c, d	}
					*/
};

/******************************************************************************/
/*!
	Static Variables
*/
/******************************************************************************/

// list of original object
static GameObj				sGameObjList[GAME_OBJ_NUM_MAX];				// Each element in this array represents a unique game object (shape)
static unsigned long		sGameObjNum;								// The number of defined game objects

// list of object instances
static GameObjInst			sGameObjInstList[GAME_OBJ_INST_NUM_MAX];	// Each element in this array represents a unique game object instance (sprite)
static unsigned long		sGameObjInstNum;							// The number of used game object instances

// pointer to the ship object
static GameObjInst *		spShip;										// Pointer to the "Ship" game object instance

// number of ship available (lives 0 = game over)
static long					sShipLives;									// The number of lives left

// the score = number of asteroid destroyed
static unsigned long		sScore;										// Current score

static bool					onValueChange;								// Whenever somethings happen
static bool					onShipHealthDecrease;						// Update console when health decrease
static bool					onAsteroidDestroyed;						// When destroying an asteroid
static bool					gameOver;									// Game is Over

static int					shipLiveIndex;								// Index to iterate through the texture index
static AEGfxTexture *		heart;										// Texture to full heart 
static AEGfxTexture *		emptyHeart;									// Texture to damaged heart
static GameObjInst	*		shipLiveTexture[SHIP_INITIAL_NUM];			// Total number of hearts

// ---------------------------------------------------------------------------

/*!**************************************************************************
\brief
	Helper function to initialise values inside the sGameObjInstList pool

\param [in] type
	Enum type of object to be created
\param [in] scale
	Size of object
\param [in] pPos
	Position of object when it's created
\param [in] pVel
	Velocity of object when it's created
\param [in] pTexture
	Texture of the object when it's created
\param [in] pColor
	Tint color of the object when it's created
\param [in] dir
	Direction of the object it will be facing when it's created

\return
	A pointer to the memory where this object is created
***************************************************************************/
GameObjInst *		gameObjInstCreate(unsigned long type, float scale, 
									  AEVec2 * pPos, AEVec2 * pVel, 
									  AEGfxTexture *texture, Color* pColor, float dir);

/*!**************************************************************************
\brief
	Helper function to "destroy" GameObjInst. It sets the active state of
	pInst to false
***************************************************************************/
void				gameObjInstDestroy(GameObjInst * pInst);

/*!**************************************************************************
\brief
	Update User's Input
***************************************************************************/
void	UpdateInput(void);

/*!**************************************************************************
\brief
	Update all game object position and transform
***************************************************************************/
void	UpdatePosition(void);

/*!**************************************************************************
\brief
	Update collision response asteroids, bullet and player
***************************************************************************/
void	UpdateCollision(void);

/*!**************************************************************************
\brief
	Update user's input to check if should restart the game
***************************************************************************/
void	Restart(void);

/*!**************************************************************************
\brief
	Helper function to spawn asteroids in the scene
***************************************************************************/
void	SpawnAsteroids(void);

/*!**************************************************************************
\brief
	Randomize an integer between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
***************************************************************************/
int		Random(int min, int max);

/*!**************************************************************************
\brief
	Randomize a float between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
***************************************************************************/
float	Random(float min, float max);

/*!**************************************************************************
\brief
	Helper function to rotate a vector by deg

\param [in, out] vec
	Result of the rotated vector by deg

\param [in] deg
	Angles in degree
***************************************************************************/
void	Rotate(AEVec2* vec, double deg);

/*!**************************************************************************
\brief
	Helper function to calculate all the bounding box of GameObjInst

\param [in, out] pInst
	GameObjInst to have it's bounding box calculated
***************************************************************************/
void	BoundingBox(GameObjInst* pInst);

/*!**************************************************************************
\brief
	Helper function to create particles behind moving ship

\param [in] dir
	Direction where the particles will move towards
***************************************************************************/
void	GenerateShipMovingParticles(float dir);

/*!**************************************************************************
\brief
	Helper function to create particles when bullet collides with asteroids

\param [in] pos
	Position of butter upon impact with asteroids
***************************************************************************/
void	GenerateExplodeParticles(AEVec2* pos);

/*!**************************************************************************
\brief
	Asteroid's load function
***************************************************************************/
void GameStateAsteroidsLoad(void)
{
	// zero the game object array
	memset(sGameObjList, 0, sizeof(GameObj) * GAME_OBJ_NUM_MAX);
	// No game objects (shapes) at this point
	sGameObjNum = 0;

	// zero the game object instance array
	memset(sGameObjInstList, 0, sizeof(GameObjInst) * GAME_OBJ_INST_NUM_MAX);
	// No game object instances (sprites) at this point
	sGameObjInstNum = 0;

	// The ship object instance hasn't been created yet, so this "spShip" pointer is initialized to 0
	spShip = nullptr;

	// load/create the mesh data (game objects / Shapes)
	GameObj * pObj;

	// =====================
	// create the ship shape
	// =====================
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_SHIP;

	AEGfxMeshStart();
	AEGfxTriAdd(
		-0.5f,  0.5f, 0xFFFF0000, 0.0f, 0.0f, 
		-0.5f, -0.5f, 0xFFFF0000, 0.0f, 0.0f,
		 0.5f,  0.0f, 0xFFFFFFFF, 0.0f, 0.0f );  

	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create ship!!");


	// =======================
	// create the bullet shape
	// =======================
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_BULLET;

	AEGfxMeshStart();
	AEGfxTriAdd(
		-0.5f,  0.5f, 0xFFFFFF00, 0.0f, 0.0f,
		-0.5f,  0.0f, 0xFFFFFF00, 0.0f, 0.0f,
		 0.5f,  0.5f, 0xFFFFFF00, 0.0f, 0.0f);

	AEGfxTriAdd(
		  0.5f,  0.5f, 0xFFFFFF00, 0.0f, 0.0f,
		 -0.5f,  0.0f, 0xFFFFFF00, 0.0f, 0.0f,
		  0.5f,  0.0f, 0xFFFFFF00, 0.0f, 0.0f);

	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create bullet!!");

	// =========================
	// create the asteroid shape
	// =========================
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_ASTEROID;

	AEGfxMeshStart();
	AEGfxTriAdd(
		-0.5f,  0.5f, 0xFF525252, 0.0f, 0.0f,
		-0.5f, -0.5f, 0xFF525252, 0.0f, 0.0f,
		 0.5f,  0.5f, 0xFF525252, 0.0f, 0.0f);

	AEGfxTriAdd(
		 0.5f,  0.5f, 0xFF525252, 0.0f, 0.0f,
		-0.5f, -0.5f, 0xFF525252, 0.0f, 0.0f,
		 0.5f, -0.5f, 0xFF525252, 0.0f, 0.0f);

	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create asteroid!!");

	// =========================
	// create the particle shape
	// =========================
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_PARTICLE_MOVING;

	AEVec2 vec1, vec2;
	AEVec2Set(&vec1, 0.5f, 0.0f);
	AEVec2Set(&vec2, 0.5f, 0.0f);
	const int tris = 20;
	const double deg = 360.0 / (double)20;
	AEGfxMeshStart();
	for (int i = 0; i <= tris; ++i)
	{
		Rotate(&vec2, deg);
		AEGfxTriAdd(vec1.x, vec1.y, 0x00FFFFFF, 0.0f, 1.0f,
					vec2.x, vec2.y, 0x00FFFFFF, 1.0f, 1.0f,
					0.0f  , 0.0f  , 0x00FFFFFF, 0.0f, 0.0f);
		AEVec2Set(&vec1, vec2.x, vec2.y);
	}
	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create moving particles!!");

	// =============================
	// create the exploding particle
	// =============================
	pObj		= sGameObjList + sGameObjNum++;
	pObj->type	= TYPE_PARTICLE_EXPLODING;

	AEGfxMeshStart();
	AEGfxTriAdd(
		-0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
		-0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
		0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

	AEGfxTriAdd(
		0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
		-0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f,
		0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 0.0f);

	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create exploding particles!!");

	// =====================
	// create the heart mesh
	// =====================
	pObj		 = sGameObjList + sGameObjNum++;
	pObj->type	 = TYPE_HEART;

	AEGfxMeshStart();
	AEGfxTriAdd(
		-0.5f, -0.5f, 0x00FFFFFF, 0.0f, 1.0f,
		 0.5f, -0.5f, 0x00FFFFFF, 1.0f, 1.0f,
		-0.5f,  0.5f, 0x00FFFFFF, 0.0f, 0.0f);

	AEGfxTriAdd(
		 0.5f, -0.5f, 0x00FFFFFF, 1.0f, 1.0f,
		 0.5f,  0.5f, 0x00FFFFFF, 1.0f, 0.0f,
		-0.5f,  0.5f, 0x00FFFFFF, 0.0f, 0.0f);

	pObj->pMesh = AEGfxMeshEnd();
	AE_ASSERT_MESG(pObj->pMesh, "fail to create heart!!");

	// Load Textures
	heart = AEGfxTextureLoad("../Resources/Textures/heart.png");
	AE_ASSERT_MESG(heart, "fail to load heart.png!!");

	emptyHeart = AEGfxTextureLoad("../Resources/Textures/empty_heart.png");
	AE_ASSERT_MESG(emptyHeart, "fail to load empty_heart.png!!");
}

/*!**************************************************************************
\brief
	Asteroid's init function
***************************************************************************/
void GameStateAsteroidsInit(void)
{
	system("cls");
	// create the main ship
	spShip = gameObjInstCreate(TYPE_SHIP, SHIP_SIZE, nullptr, nullptr, nullptr, nullptr, 0.0f);
	AE_ASSERT(spShip);	

	AEVec2 heartPos;
	AEVec2Set(&heartPos, -375.0f, 275.0f);

	for (unsigned int i = 0; i < SHIP_INITIAL_NUM; ++i)
		*(shipLiveTexture + i) = gameObjInstCreate(TYPE_HEART, HEART_SIZE, &AEVec2({ heartPos.x + 50.0f * i, heartPos.y }), nullptr, heart, nullptr, 0.0f);

	shipLiveIndex = SHIP_INITIAL_NUM - 1;

	// CREATE THE INITIAL ASTEROIDS INSTANCES USING THE "gameObjInstCreate" FUNCTION
	gameObjInstCreate(TYPE_ASTEROID, 25.0f,  &AEVec2({  275.0f, -175.0f }),  &AEVec2({  cosf(45.0f)  * 150.0f,  sinf(45.0f)  *  75.0f }), nullptr, nullptr, 0.0f);
	gameObjInstCreate(TYPE_ASTEROID, 50.0f,  &AEVec2({ -250.0f,  150.0f }),  &AEVec2({  cosf(-45.0f) * 100.0f,  sinf(-45.0f) * 50.0f }), nullptr, nullptr, 0.0f);
	gameObjInstCreate(TYPE_ASTEROID, 75.0f,  &AEVec2({ -150.0f, -275.0f }),  &AEVec2({  cosf(135.0f) * 50.0f ,  sinf(135.0f) * 125.0f }), nullptr, nullptr, 0.0f);
	gameObjInstCreate(TYPE_ASTEROID, 100.0f, &AEVec2({  350.0f,  200.0f }),  &AEVec2({  cosf(90.0f)  * 20.0f ,  sinf(90.0f)  *  150.0f }), nullptr, nullptr, 0.0f);

	// reset the score and the number of ships
	sScore					= 0;
	sShipLives				= SHIP_INITIAL_NUM;

	onValueChange			= true;
	onShipHealthDecrease	= true;
	onAsteroidDestroyed		= true;
	gameOver				= false;
}

/*!**************************************************************************
\brief
	Asteroid's update function
***************************************************************************/
void GameStateAsteroidsUpdate(void)
{
	UpdateInput();
	UpdateCollision();
	UpdatePosition();
	Restart();
}

/*!**************************************************************************
\brief
	Asteroids draw function
***************************************************************************/
void GameStateAsteroidsDraw(void)
{
	char strBuffer[1024];
	
	// draw all object instances in the list
	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst * pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;
		
		switch (pInst->pObject->type)
		{
			case TYPE_HEART:

			{
				AEGfxSetRenderMode(AE_GFX_RM_TEXTURE);
				AEGfxSetBlendMode(AE_GFX_BM_BLEND);
				break;
			}
			default:
			{
				AEGfxSetRenderMode(AE_GFX_RM_COLOR);
				AEGfxSetBlendMode(AE_GFX_BM_NONE);
				break;
			}
		}

		AEGfxSetPosition(pInst->posCurr.x, pInst->posCurr.y);
		AEGfxSetTintColor(pInst->color.r, pInst->color.g, pInst->color.b, pInst->color.a);
		AEGfxSetTransparency(pInst->color.a);
		AEGfxTextureSet(pInst->texture, 0, 0);
		AEGfxSetTransform(pInst->transform.m);
		AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
	}

	//You can replace this condition/variable by your own data.
	//The idea is to display any of these variables/strings whenever a change in their value happens
	if(onValueChange)
	{
		if (onAsteroidDestroyed)
		{
			sprintf_s(strBuffer, "Score: %d", sScore);
			//AEGfxPrint(10, 10, (u32)-1, strBuffer);
			printf("%s \n", strBuffer);
		}

		if (onShipHealthDecrease)
		{
			sprintf_s(strBuffer, "Ship Left: %d", sShipLives >= 0 ? sShipLives : 0);
			//AEGfxPrint(600, 10, (u32)-1, strBuffer);
			printf("%s \n", strBuffer);
		}

		// display the game over message
		if (sShipLives <= 0)
		{
			//AEGfxPrint(280, 260, 0xFFFFFFFF, "       GAME OVER       ");
			printf("       GAME OVER       \n");
			gameOver = true;
		}

		if (SCORE_TO_WIN <= sScore)
		{
			printf("       YOU ROCK!!       \n");
			gameOver = true;
		}

		onValueChange			= false;
		onShipHealthDecrease	= false;
		onAsteroidDestroyed		= false;
	}

	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	sprintf_s(strBuffer, "Score: %u", sScore);
	f32 TextWidth, TextHeight;
	AEGfxGetPrintSize(fontID, strBuffer, 1.0f, TextWidth, TextHeight);
	AEGfxPrint(fontID, strBuffer, 0.95f - TextWidth, 0.95f - TextHeight, 1.0f, 1.f, 1.f, 1.f);

	if (gameOver)
	{
		if (SCORE_TO_WIN <= sScore)
		{
			sprintf_s(strBuffer, "YOU ROCK!!");
			AEGfxPrint(fontID, strBuffer, -0.025f, 0.0f, 1.0f, 1.f, 1.f, 1.f);
			sprintf_s(strBuffer, "Press 'R' to restart");
			AEGfxPrint(fontID, strBuffer, -0.05f, -0.05f - TextHeight, 1.0f, 1.f, 1.f, 1.f);
		}
		else if(0 >= sShipLives)
		{
			sprintf_s(strBuffer, "GAME OVER!!");
			AEGfxPrint(fontID, strBuffer, -0.025f, 0.0f, 1.0f, 1.f, 1.f, 1.f);
			sprintf_s(strBuffer, "Press 'R' to restart");
			AEGfxPrint(fontID, strBuffer, -0.05f, -0.05f - TextHeight, 1.0f, 1.f, 1.f, 1.f);
		}
	}
}

/*!**************************************************************************
\brief
	Asteroid's free function
***************************************************************************/
void GameStateAsteroidsFree(void)
{
	// kill all object instances in the array using "gameObjInstDestroy"
	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
		gameObjInstDestroy(sGameObjInstList + i);
}

/*!**************************************************************************
\brief
	Asteroid's unload function
***************************************************************************/
void GameStateAsteroidsUnload(void)
{
	// free all mesh data (shapes) of each object using "AEGfxTriFree"
	for (unsigned long i = 0; i < GAME_OBJ_NUM_MAX; ++i)
	{
		if ((sGameObjList + i)->pMesh)
			AEGfxMeshFree((sGameObjList + i)->pMesh);
	}
	AEGfxTextureUnload(heart);
	AEGfxTextureUnload(emptyHeart);
}

/*!**************************************************************************
\brief
	Helper function to initialise values inside the sGameObjInstList pool

\param [in] type
	Enum type of object to be created
\param [in] scale
	Size of object
\param [in] pPos
	Position of object when it's created
\param [in] pVel
	Velocity of object when it's created
\param [in] pTexture
	Texture of the object when it's created
\param [in] pColor
	Tint color of the object when it's created
\param [in] dir
	Direction of the object it will be facing when it's created

\return
	A pointer to the memory where this object is created
***************************************************************************/
GameObjInst * gameObjInstCreate(unsigned long type, 
								float scale, 
								AEVec2 * pPos, 
								AEVec2 * pVel,
								AEGfxTexture* pTexture,
								Color * pColor,
								float dir)
{
	AEVec2 zero;
	AEVec2Zero(&zero);

	AE_ASSERT_PARM(type < sGameObjNum);
	
	// loop through the object instance list to find a non-used object instance
	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst * pInst = sGameObjInstList + i;

		// check if current instance is not used
		if (pInst->flag == 0)
		{
			// it is not used => use it to create the new instance
			pInst->pObject	= sGameObjList + type;
			pInst->flag		= FLAG_ACTIVE;
			pInst->scale	= scale;
			pInst->posCurr	= pPos ? *pPos : zero;
			pInst->velCurr	= pVel ? *pVel : zero;
			pInst->dirCurr	= dir;
			pInst->color	= pColor ? *pColor : Color({ 1.0f, 1.0f, 1.0f,1.0f });
			pInst->texture	= pTexture;
			
			// return the newly created instance
			return pInst;
		}
	}

	// cannot find empty slot => return 0
	return 0;
}

/*!**************************************************************************
\brief
	Helper function to "destroy" GameObjInst. It sets the active state of
	pInst to false
***************************************************************************/
void gameObjInstDestroy(GameObjInst * pInst)
{
	// if instance is destroyed before, just return
	if (pInst->flag == 0)
		return;

	// zero out the flag
	pInst->flag = 0;
}

/*!**************************************************************************
\brief
	Update User's Input
***************************************************************************/
void UpdateInput(void)
{
	if (gameOver)
		return;

	if (AEInputCheckCurr(AEVK_UP))
	{
		AEVec2 acc;
		AEVec2Set(&acc, cosf(spShip->dirCurr), sinf(spShip->dirCurr));
		AEVec2Scale(&acc, &acc, SHIP_ACCEL_FORWARD * g_dt);
		AEVec2Add(&spShip->velCurr, &spShip->velCurr, &acc);
		AEVec2Scale(&spShip->velCurr, &spShip->velCurr, VELOCITY_CAP_PERCENT);
		GenerateShipMovingParticles(-1.0f);
	}

	if (AEInputCheckCurr(AEVK_DOWN))
	{
		AEVec2 acc;
		AEVec2Set(&acc, -cosf(spShip->dirCurr), -sinf(spShip->dirCurr));
		AEVec2Scale(&acc, &acc, SHIP_ACCEL_FORWARD * g_dt);
		AEVec2Add(&spShip->velCurr, &spShip->velCurr, &acc);
		GenerateShipMovingParticles(1.0f);
	}

	if (AEInputCheckCurr(AEVK_LEFT))
	{
		spShip->dirCurr += SHIP_ROT_SPEED * g_dt;
		spShip->dirCurr = AEWrap(spShip->dirCurr, -PI, PI);
	}

	if (AEInputCheckCurr(AEVK_RIGHT))
	{
		spShip->dirCurr -= SHIP_ROT_SPEED * g_dt;
		spShip->dirCurr = AEWrap(spShip->dirCurr, -PI, PI);
	}

	if (AEInputCheckTriggered(AEVK_SPACE))
	{
		AEVec2 vel;
		AEVec2Set(&vel, cosf(spShip->dirCurr), sinf(spShip->dirCurr));
		AEVec2Scale(&vel, &vel, BULLET_SPEED);
		gameObjInstCreate(TYPE_BULLET, BULLET_SIZE, &spShip->posCurr, &vel, nullptr, nullptr, spShip->dirCurr);
	}
}

/*!**************************************************************************
\brief
	Update all game object position and transform
***************************************************************************/
void UpdatePosition(void)
{
	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst* pInst = sGameObjInstList + i;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		AEVec2 vel;
		AEVec2Scale(&vel, &pInst->velCurr, g_dt);
		AEVec2Add(&pInst->posCurr, &pInst->posCurr, &vel);

		// check if the object is a ship
		switch (pInst->pObject->type)
		{
			case TYPE_SHIP:
			{
				// warp the ship from one end of the screen to the other
				pInst->posCurr.x = AEWrap(pInst->posCurr.x, AEGfxGetWinMinX() - SHIP_SIZE,
															AEGfxGetWinMaxX() + SHIP_SIZE);
				pInst->posCurr.y = AEWrap(pInst->posCurr.y, AEGfxGetWinMinY() - SHIP_SIZE,
															AEGfxGetWinMaxY() + SHIP_SIZE);
				break;
			}
			// Remove bullets that go out of bounds
			case TYPE_BULLET:
			{
				AEVec2 pos = pInst->posCurr;
				if (pos.x < AEGfxGetWinMinX() - BULLET_SIZE || pos.x > AEGfxGetWinMaxX() + BULLET_SIZE || 
					pos.y < AEGfxGetWinMinY() - BULLET_SIZE || pos.y > AEGfxGetWinMaxY() + BULLET_SIZE)
					gameObjInstDestroy(pInst);
				break;
			}
			// Wrap asteroids here
			case TYPE_ASTEROID:
			{
				pInst->posCurr.x = AEWrap(pInst->posCurr.x, AEGfxGetWinMinX() - pInst->scale,
															AEGfxGetWinMaxX() + pInst->scale);
				pInst->posCurr.y = AEWrap(pInst->posCurr.y, AEGfxGetWinMinY() - pInst->scale,
															AEGfxGetWinMaxY() + pInst->scale);
				break;
			}
			case TYPE_PARTICLE_MOVING:
			{
				const float RANDOM_SCALE = Random(1.0f, 2.0f);
				const float RAD = 20.0f * PI / 180.0f;
				pInst->dirCurr += RAD * RANDOM_SCALE;
				if (pInst->dirCurr >= PI * PARTICLE_TIME_SCALE)
					gameObjInstDestroy(pInst);
				break;
			}
			case TYPE_PARTICLE_EXPLODING:
			{
				const float RANDOM_SCALE = Random(1.0f, 2.0f);
				pInst->scale += PARTICLE_SCALE_INCREASE * RANDOM_SCALE;
				if (30.0f <= pInst->scale)
					gameObjInstDestroy(pInst);
				break;
			}
		}
	}

	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++)
	{
		GameObjInst* pInst = sGameObjInstList + i;
		AEMtx33		 trans, rot, scale;

		// skip non-active object
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;

		AEMtx33Trans(&trans, pInst->posCurr.x, pInst->posCurr.y);
		AEMtx33Rot(&rot, pInst->dirCurr);
		AEMtx33Scale(&scale, pInst->scale, pInst->scale);
		AEMtx33Concat(&pInst->transform, &rot, &scale);
		AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);

		UNREFERENCED_PARAMETER(trans);
		UNREFERENCED_PARAMETER(rot);
		UNREFERENCED_PARAMETER(scale);
	}
}

/*!**************************************************************************
\brief
	Update collision response asteroids, bullet and player
***************************************************************************/
void UpdateCollision(void)
{
	if (gameOver)
		return;

	for (unsigned int i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst* pInst = sGameObjInstList + i;
		if ((pInst->flag & FLAG_ACTIVE) == 0)
			continue;
		BoundingBox(pInst);
	}

	for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; ++i)
	{
		GameObjInst* oi1 = sGameObjInstList + i;
		// skip non-active object
		if ((oi1->flag & FLAG_ACTIVE) == 0 || oi1->pObject->type != TYPE_ASTEROID)
			continue;

		for (unsigned long j = 0; j < GAME_OBJ_INST_NUM_MAX; ++j)
		{
			GameObjInst* oi2 = sGameObjInstList + j;
			if ((oi2->flag & FLAG_ACTIVE) == 0 || oi2->pObject->type >= TYPE_ASTEROID || oi1 == oi2)
				continue;

			if (!CollisionIntersection_RectRect(oi1->boundingBox, oi1->velCurr, oi2->boundingBox, oi2->velCurr))
				continue;
			switch (oi2->pObject->type)
			{
				case TYPE_SHIP:
				{
					// deduct hp
					GenerateExplodeParticles(&oi2->posCurr);
					--sShipLives;
					if (0 < sShipLives)
					{
						AEVec2Set(&oi2->posCurr, 0.0f, 0.0f);	// reset position
						AEVec2Set(&oi2->velCurr, 0.0f, 0.0f);	// reset velocity
						SpawnAsteroids();
					}
					else
						gameObjInstDestroy(oi2);
					onShipHealthDecrease = true;
					shipLiveTexture[shipLiveIndex--]->texture = emptyHeart;
					break;
				}
				case TYPE_BULLET:
				{
					// add points
					sScore += SCORE_PER_DESTROY;
					SpawnAsteroids();
					GenerateExplodeParticles(&oi2->posCurr);
					gameObjInstDestroy(oi2);
					onAsteroidDestroyed = true;
					break;
				}
			}
			gameObjInstDestroy(oi1);
			onValueChange = true;
		}
	}
}

/*!**************************************************************************
\brief
	Helper function to spawn asteroids in the scene
***************************************************************************/
void SpawnAsteroids(void)
{
	const int SPAWNS = Random(1, 2);
	for (int i = 0; i < SPAWNS; ++i)
	{
		const float RAD		= Random(-PI, PI);
		const float VEL_X	= Random(ASTEROID_MIN_VELOCITY, ASTEROID_MAX_VELOCITY);
		const float VEL_Y	= Random(ASTEROID_MIN_VELOCITY, ASTEROID_MAX_VELOCITY);
		const int SIZE		= Random(ASTEROID_MIN_SIZE, ASTEROID_MAX_SIZE);

		const int SPAWN_POS = Random(1, 4);
		AEVec2 pos;
		switch (SPAWN_POS)
		{
			case 1:
			{
				AEVec2Set(&pos, 0.0f, AEGfxGetWinMaxY() + SIZE);
				break;
			}
			case 2:
			{
				AEVec2Set(&pos, AEGfxGetWinMaxX() + SIZE, 0.0f);
				break;
			}
			case 3:
			{
				AEVec2Set(&pos, 0.0f, AEGfxGetWinMinY() - SIZE);
				break;
			}
			case 4:
			{
				AEVec2Set(&pos, AEGfxGetWinMinX() - SIZE, 0.0f);
				break;
			}
		}
		GameObjInst* pInst = gameObjInstCreate(TYPE_ASTEROID, SIZE, &pos, 
							 &AEVec2({ cosf(RAD) * VEL_X, sinf(RAD) * VEL_Y }),
							 nullptr, nullptr, 0.0f);
		BoundingBox(pInst);
	}
}

/*!**************************************************************************
\brief
	Update user's input to check if should restart the game
***************************************************************************/
void Restart(void)
{
	if (!gameOver)
		return;

	if (AEInputCheckTriggered(AEVK_R))
	{
		gGameStateNext = GS_RESTART;
	}
}

/*!**************************************************************************
\brief
	Helper function to rotate a vector by deg

\param [in, out] vec
	Result of the rotated vector by deg

\param [in] deg
	Angles in degree
***************************************************************************/
void Rotate(AEVec2* vec, double deg)
{
	const double rad = deg * PI / 180.0;
	const double sinTheta = sin(rad);
	const double cosTheta = cos(rad);
	Matrix2x2 rotation;
	rotation.m[0][0] = cosTheta;	// a
	rotation.m[0][1] = -sinTheta;	// b
	rotation.m[1][0] = sinTheta;	// c
	rotation.m[1][1] = cosTheta;	// d

	const float x = rotation.m[0][0] * vec->x + rotation.m[0][1] * vec->y;
	const float y = rotation.m[1][0] * vec->x + rotation.m[1][1] * vec->y;
	AEVec2Set(vec, x, y);
}

/*!**************************************************************************
\brief
	Randomize an integer between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
***************************************************************************/
int Random(int min, int max)
{
	return rand() % max + min;
}

/*!**************************************************************************
\brief
	Randomize a float between min and max

\param [in] min
	Lower bound of random number

\param [in] max
	Upper bound of random number

\return
	Pseudo-random number between min and max
***************************************************************************/
float Random(float min, float max)
{
	return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

/*!**************************************************************************
\brief
	Helper function to calculate all the bounding box of GameObjInst

\param [in, out] pInst
	GameObjInst to have it's bounding box calculated
***************************************************************************/
void BoundingBox(GameObjInst* pInst)
{
	if (!pInst)
		return;
	pInst->boundingBox.min.x = pInst->posCurr.x - pInst->scale * 0.5f;
	pInst->boundingBox.max.x = pInst->posCurr.x + pInst->scale * 0.5f;
	pInst->boundingBox.min.y = pInst->posCurr.y - pInst->scale * 0.5f;
	pInst->boundingBox.max.y = pInst->posCurr.y + pInst->scale * 0.5f;
}

/*!**************************************************************************
\brief
	Helper function to create particles behind moving ship

\param [in] dir
	Direction where the particles will move towards
***************************************************************************/
void GenerateShipMovingParticles(float dir)
{
	Color red	 = Color({ 1.0f, 0.0f,  0.0f, 1.0f });
	Color orange = Color({ 1.0f, 0.65f, 0.0f, 1.0f });
	Color yellow = Color({ 1.0f, 1.0f,  0.0f, 1.0f });

	const float RAD					= ANGLE_OF_PARTICLE_SPAWN * PI / 180.0;
	const int	PARTICLES_SPAWN		= Random(10, 20);
	int			particleSpawnIndex = 0;

	bool gotParticles = true;
	for (int i = 0; i < PARTICLES_SPAWN; ++i)
	{
		AEVec2 vel;
		const float ANGLE	= Random(spShip->dirCurr - RAD, spShip->dirCurr + RAD);
		AEVec2Set(&vel, cosf(ANGLE) * SPEED_MOVING_PARTICLE * dir, sinf(ANGLE) * SPEED_MOVING_PARTICLE * dir);

		const float SIZE	= Random(SIZE_MOVING_PARTICLE_MIN, SIZE_MOVING_PARTICLE_MAX);

		Color color;
		const int COLOR		= Random(1, 3);
		switch (COLOR)
		{
			case 1:
			{
				color = red;
				break;
			}
			case 2:
			{
				color = orange;
				break;
			}
			case 3:
			{
				color = yellow;
				break;
			}
		}
		gameObjInstCreate(TYPE_PARTICLE_MOVING, SIZE, &AEVec2({spShip->posCurr.x - (cosf(spShip->dirCurr) * spShip->scale) * 0.5f, spShip->posCurr.y - (sinf(spShip->dirCurr) * spShip->scale) * 0.5f }), &vel, nullptr, &color, 0.0f);
	}
}

/*!**************************************************************************
\brief
	Helper function to create particles when bullet collides with asteroids

\param [in] pos
	Position of butter upon impact with asteroids
***************************************************************************/
void GenerateExplodeParticles(AEVec2* pos)
{
	// If position is nullptr or both flags are true
	if (!pos)
		return;

	Color red = Color({ 1.0f, 0.0f,  0.0f, 1.0f });
	Color orange = Color({ 1.0f, 0.65f, 0.0f, 1.0f });
	Color yellow = Color({ 1.0f, 1.0f,  0.0f, 1.0f });

	const int SPAWNS = Random(3, 5);
	for (int i = 0; i < SPAWNS; ++i)
	{
		Color color;
		const int COLOR = Random(1, 3);
		switch (COLOR)
		{
			case 1:
			{
				color = red;
				break;
			}
			case 2:
			{
				color = orange;
				break;
			}
			case 3:
			{
				color = yellow;
				break;
			}
		}
		const float SCALE = Random(15.0f, 20.0f);
		const float x = Random(0.0f, 20.0f);
		const float y = Random(0.0f, 20.0f);
		int sign = Random(1, 2);
		const float x_sign = sign == 1 ? 1.0f : -1.0f;
		sign = Random(1, 2);
		const float y_sign = sign == 1 ? 1.0f : -1.0f;

		gameObjInstCreate(TYPE_PARTICLE_EXPLODING, SCALE, &AEVec2({ pos->x + x * x_sign, pos->y + y * y_sign }), nullptr, nullptr, &color, 0.0f);
	}
}