#pragma once
#include "BehaviorNode.h"

class L_ItsMyTurn : public BaseNode<L_ItsMyTurn>
{
public:
    L_ItsMyTurn();

protected:
    float timer;

    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};