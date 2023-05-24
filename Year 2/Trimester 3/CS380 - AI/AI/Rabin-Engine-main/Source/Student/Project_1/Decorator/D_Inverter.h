#pragma once
#include "BehaviorNode.h"

class D_Inverter : public BaseNode<D_Inverter>
{
public:
    D_Inverter();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};