#pragma once
#include "BehaviorNode.h"

class L_RandomPointOutsideStore : public BaseNode<L_RandomPointOutsideStore>
{
private:
    Vec3 targetPosition{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};