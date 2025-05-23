/******************************************************************************/
/*!
\file		pch.cpp
\project	CS380/CS580 AI Framework
\author		Dustin Holmes
\summary	Pre-compiled header

Copyright (C) 2018 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#pragma once

#include <WinSDKVer.h>
#define _WIN32_WINNT 0x0602

#include <SDKDDKVer.h>
#define NOMINMAX
#include <dwrite_2.h>
#include <d2d1_2.h>
// Use the C++ standard templated min/max


// DirectX apps don't need GDI
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP

// Include <mcx.h> if you need this
#define NOMCX

// Include <winsvc.h> if you need this
#define NOSERVICE

// WinHelp is deprecated
#define NOHELP

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#include <wrl/client.h>


#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <DirectXMath.h>
#include <DirectXColors.h>



#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "CommonStates.h"
#include "DDSTextureLoader.h"
#include "DirectXHelpers.h"
#include "Effects.h"
#include "GamePad.h"
#include "GeometricPrimitive.h"
#include "GraphicsMemory.h"
#include "Keyboard.h"
#include "Model.h"
#include "Mouse.h"
#include "PostProcess.h"
#include "PrimitiveBatch.h"
#include "ScreenGrab.h"
#include "SimpleMath.h"
#include "SpriteBatch.h"
#include "SpriteFont.h"
#include "VertexTypes.h"
#include "WICTextureLoader.h"

#include "Global.h"
#include "Core/Engine.h"
#include "Terrain/Terrain.h"
#include "Rendering/DeviceResources.h"
#include "Rendering/SimpleRenderer.h"
#include "Rendering/MeshRenderer.h"
#include "Rendering/TextRenderer.h"
#include "Rendering/UISpriteRenderer.h"
#include "Rendering/DebugRenderer.h"
#include "Agent/AgentOrganizer.h"
#include "Projects/Project.h"
#include "Input/InputHandler.h"
#include "UI/UICoordinator.h"
#include "Core/Messenger.h"
#include "Misc/RNG.h"
#include "Core/Serialization.h"
#include "Core/AudioManager.h"
#include "../Source/Student/Project_1/MyVar.h"

namespace DX
{
    inline void ThrowIfFailed(HRESULT hr)
    {
        if (FAILED(hr))
        {
            // Set a breakpoint on this line to catch DirectX API errors
            throw std::exception();
        }
    }
}