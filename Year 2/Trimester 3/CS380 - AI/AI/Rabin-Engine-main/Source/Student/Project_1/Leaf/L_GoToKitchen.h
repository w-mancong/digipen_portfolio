#pragma once
#include "BehaviorNode.h"

class L_GoToKitchen : public BaseNode<L_GoToKitchen>
{
public:
    L_GoToKitchen();

    Vec3 targetPosition{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};