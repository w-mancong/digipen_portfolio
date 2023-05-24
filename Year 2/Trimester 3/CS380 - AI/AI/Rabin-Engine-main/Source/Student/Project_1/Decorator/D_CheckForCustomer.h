#pragma once
#include "BehaviorNode.h"

class D_CheckForCustomer : public BaseNode<D_CheckForCustomer>
{
public:
    D_CheckForCustomer();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};