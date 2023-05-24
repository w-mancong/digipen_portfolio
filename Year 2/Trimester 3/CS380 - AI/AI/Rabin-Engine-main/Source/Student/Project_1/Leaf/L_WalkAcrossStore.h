#pragma once
#include "BehaviorNode.h"

class L_WalkAcrossStore : public BaseNode<L_WalkAcrossStore>
{
private:
    Vec3 targetPosition{};
    float delay{}, timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};