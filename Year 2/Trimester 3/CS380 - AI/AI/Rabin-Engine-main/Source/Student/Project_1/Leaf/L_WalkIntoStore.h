#pragma once
#include "BehaviorNode.h"

class L_WalkIntoStore : public BaseNode<L_WalkIntoStore>
{
private:
    bool atEntrance{ false };
    Vec3 targetPosition[2]{};
    size_t index{};
    float delay{}, timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};