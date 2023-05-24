#pragma once
#include "BehaviorNode.h"

class L_WaitForChef : public BaseNode<L_WaitForChef>
{
public:
    L_WaitForChef();

private:
    float timer{}, angle{}, flag{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};