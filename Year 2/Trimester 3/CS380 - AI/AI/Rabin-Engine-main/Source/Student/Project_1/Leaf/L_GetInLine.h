#pragma once
#include "BehaviorNode.h"

class L_GetInLine : public BaseNode<L_GetInLine>
{
private:
    Vec3 targetPosition{};
    float delay{}, timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};