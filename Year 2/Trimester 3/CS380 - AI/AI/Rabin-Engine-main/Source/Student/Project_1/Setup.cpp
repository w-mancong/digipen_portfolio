#include <pch.h>
#include "Projects/ProjectOne.h"
#include "Agent/CameraAgent.h"

void ProjectOne::setup()
{
    // Create cashier
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Cashier", BehaviorTreeTypes::Cashier);
        ba->set_scaling(2.0f);
        ba->set_position({MyVar::CELL_SIZE * 13.0f + MyVar::HALF_CELL_SIZE, 0.0f, MyVar::CELL_SIZE * 3.0f});
        ba->set_color({ 0.93f, 0.88f, 0.19f });
        ba->set_yaw(MyVar::DegToRad(-90.0f));

        Blackboard& bb = ba->get_blackboard();
        bb.set_value("queueIndex", LEFT_Q);
        bb.set_value("originalPosition", ba->get_position());
    }

    // Create left chef
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Chef", BehaviorTreeTypes::Chef);
        ba->set_scaling(2.0f);
        ba->set_position({ MyVar::CELL_SIZE * 18.0f + MyVar::HALF_CELL_SIZE, 0.0f, MyVar::CELL_SIZE * 3.0f });
        ba->set_color({ 0.28f, 0.46f, 0.96f });
        ba->set_yaw(MyVar::DegToRad(90.0f));

        Blackboard& bb = ba->get_blackboard();
        bb.set_value("queueIndex", LEFT_Q);
        bb.set_value("originalPosition", ba->get_position());
        bb.set_value("originalColor", ba->get_color());
    }

    // Create Manager
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Manager", BehaviorTreeTypes::Manager);
        ba->set_scaling(2.0f);
        ba->set_position({ MyVar::CELL_SIZE * 13.0f + MyVar::HALF_CELL_SIZE, 0.0f, MyVar::CELL_SIZE * 7.0f });
        ba->set_color({ 0.61f, 0.16f, 0.81f });
        ba->set_yaw(MyVar::DegToRad(-90.0f));

        Blackboard& bb = ba->get_blackboard();
        bb.set_value("queueIndex", RIGHT_Q);
        bb.set_value("originalPosition", ba->get_position());
    }

    // Create right chef
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Chef", BehaviorTreeTypes::Chef);
        ba->set_scaling(2.0f);
        ba->set_position({ MyVar::CELL_SIZE * 18.0f + MyVar::HALF_CELL_SIZE, 0.0f, MyVar::CELL_SIZE * 7.0f });
        ba->set_color({ 0.28f, 0.46f, 0.96f });
        ba->set_yaw(MyVar::DegToRad(90.0f));

        Blackboard& bb = ba->get_blackboard();
        bb.set_value("queueIndex", RIGHT_Q);
        bb.set_value("originalPosition", ba->get_position());
        bb.set_value("originalColor", ba->get_color());
    }

    // Create customer spawner
    agents->create_behavior_agent("CustomerSpawner", BehaviorTreeTypes::CustomerSpawner)->set_scaling(0.0f);

    //CreateCustomers(20);

    // you can technically load any map you want, even create your own map file,
    // but behavior agents won't actually avoid walls or anything special, unless you code that yourself
    // that's the realm of project 2 though
    terrain->goto_map(0);

    // you can also enable the pathing layer and set grid square colors as you see fit
    // works best with map 0, the completely blank map
    CreateMapLayout();

    // camera position can be modified from this default as well
    auto camera = agents->get_camera_agent();
    camera->set_position(Vec3(terrain->mapSizeInWorld * 0.5f - 12.5f, 185.0f, terrain->mapSizeInWorld * 0.5f));
    float rad = MyVar::DegToRad(89.9f);
    camera->set_pitch(rad); // 35 degrees

    audioManager->SetVolume(0.5f);
    audioManager->PlaySoundEffect(L"Assets\\Audio\\retro.wav");
    // uncomment for example on playing music in the engine (must be .wav)
    // audioManager->PlayMusic(L"Assets\\Audio\\motivate.wav");
    // audioManager->PauseMusic(...);
    // audioManager->ResumeMusic(...);
    // audioManager->StopMusic(...);
}

void ProjectOne::CreateMapLayout(void)
{
    int constexpr MAX_CELLS{ 20 }, KITCHEN_HEIGHT{ 4 }, CASHIER_AREA_HEIGHT{ 2 };
    terrain->pathLayer.set_enabled(true);
    // kitchen
    {
        for (int i{}; i < KITCHEN_HEIGHT; ++i)
        {
            for(int j{}; j < (MAX_CELLS >> 1); ++j)
                terrain->pathLayer.set_value(MAX_CELLS - 1 - i, j, Color(0.98f, 0.78f, 0.08f));
        }
    }

    // cashier area
    {
        int constexpr GAP{ 4 }; // gap between kitchen and cashier
        for (int i{}; i < CASHIER_AREA_HEIGHT; ++i)
        {
            for(int j{}; j < (MAX_CELLS >> 1); ++j)
                terrain->pathLayer.set_value(MAX_CELLS - GAP - KITCHEN_HEIGHT - i, j, Colors::Black);
        }
    }

    // entrance
    terrain->pathLayer.set_value(0, 9, Color(0.44f, 1.0f, 0.086f));
    terrain->pathLayer.set_value(0, 10, Color(0.44f, 1.0f, 0.086f));

    // tables
    {
        size_t index = 0;
        memset(MyVar::tables, 0, sizeof(MyVar::tables));
        for (size_t i{}; i < sizeof(MyVar::tables) / sizeof(*MyVar::tables); ++i)
            (MyVar::tables + i)->isEmpty = true;

        auto MakeTable = [&index](int x, int z)
        {
            for (int i{}; i < 2; ++i)
            {
                for (int j{}; j < 3; ++j)
                    terrain->pathLayer.set_value(x - i, z + j, Color(0.46f, 0.41f, 0.23f));
            }
            MyVar::tables[index++].position = Vec3( x * MyVar::CELL_SIZE, 0.0f, (z + 1) * MyVar::CELL_SIZE + MyVar::HALF_CELL_SIZE );
        };

                           MakeTable(17, 16);   // writing it this way as it mimics the table layout
        MakeTable(12, 12); MakeTable(12, 16);   // when the program is running
        MakeTable(7 , 12); MakeTable(7 , 16);
        MakeTable(2 , 12); MakeTable(2 , 16);   
    }
}

void ProjectOne::CreateCustomers(size_t numberOfCustomers)
{
    for (size_t i{}; i < numberOfCustomers; ++i)
    {
        BehaviorAgent* ba = agents->create_behavior_agent("Customer", BehaviorTreeTypes::Customer);
        bool spawnLeft = RNG::coin_toss();
        float offset = RNG::range(-30.0f, 30.0f);
        if (spawnLeft)
            ba->set_position(Vec3(-10.0f, 0.0f, -40.0f + offset));
        else
            ba->set_position(Vec3(-10.0f, 0.0f, 165.0f + offset));
        ba->set_scaling(2.0f);
        ba->set_movement_speed(RNG::range(10.0f, 20.0f));
        ba->get_blackboard().set_value("ordered", false);
        ba->get_blackboard().set_value("leftQueue", false);
        ba->get_blackboard().set_value("leavingStore", false);
        ba->get_blackboard().set_value("spawnLeft", spawnLeft);
        ba->get_blackboard().set_value("queueID", std::numeric_limits<int>::max());
        ba->get_blackboard().set_value("queueIndex", std::numeric_limits<size_t>::max());
    }
}
