#pragma once
#include "BehaviorNode.h"

class L_WaitForCustomer : public BaseNode<L_WaitForCustomer>
{
public:
    L_WaitForCustomer();

private:
    float timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};