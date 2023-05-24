#pragma once
#include "BehaviorNode.h"

class L_MoveTowardsCashier : public BaseNode<L_MoveTowardsCashier>
{
private:
    Vec3 targetPosition{}, actualPosition{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};