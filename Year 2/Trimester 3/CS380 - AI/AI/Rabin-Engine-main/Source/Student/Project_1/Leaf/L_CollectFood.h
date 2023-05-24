#pragma once
#include "BehaviorNode.h"

class L_CollectFood : public BaseNode<L_CollectFood>
{
public:
    L_CollectFood();

protected:
    float timer;

    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};