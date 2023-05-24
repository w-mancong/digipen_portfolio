#pragma once
#include "BehaviorNode.h"

class L_LeaveStore : public BaseNode<L_LeaveStore>
{
public:
    L_LeaveStore();

private:
    int counter = 0;

protected:
    Vec3 targetPosition{};

    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};