#pragma once
#include "BehaviorNode.h"

class L_WalkToTable : public BaseNode<L_WalkToTable>
{
private:
    size_t counter = 0;
    Vec3 targetPosition{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};