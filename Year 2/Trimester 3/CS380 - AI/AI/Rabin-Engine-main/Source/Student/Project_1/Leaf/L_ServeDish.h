#pragma once
#include "BehaviorNode.h"

class L_ServeDish : public BaseNode<L_ServeDish>
{
public:
    L_ServeDish();

private:
    Vec3 targetPosition{};
    int counter = 0;

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};