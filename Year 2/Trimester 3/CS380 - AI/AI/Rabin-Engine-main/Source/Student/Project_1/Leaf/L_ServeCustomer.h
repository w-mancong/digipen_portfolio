#pragma once
#include "BehaviorNode.h"

class L_ServeCustomer : public BaseNode<L_ServeCustomer>
{
public:
    L_ServeCustomer();

private:
    Vec3 targetPosition{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};