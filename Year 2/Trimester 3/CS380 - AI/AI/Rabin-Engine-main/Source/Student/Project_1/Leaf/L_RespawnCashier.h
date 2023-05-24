#pragma once
#include "BehaviorNode.h"

class L_RespawnCashier : public BaseNode<L_RespawnCashier>
{
public:
    L_RespawnCashier();

protected:
    float timer{}, angle{}, scale{};

    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};