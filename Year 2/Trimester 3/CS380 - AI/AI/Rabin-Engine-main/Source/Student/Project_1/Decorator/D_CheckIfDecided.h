#pragma once
#include "BehaviorNode.h"

class D_CheckIfDecided : public BaseNode<D_CheckIfDecided>
{
public:
    D_CheckIfDecided();

private:
    bool decisions[2]{ false, false }; // 0: boolean to see if agent have came into this node already
                                       // 1: boolean to check if agent wanna go into store

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};