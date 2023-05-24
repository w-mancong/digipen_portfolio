#pragma once
#include "BehaviorNode.h"

class L_WaitForOrder : public BaseNode<L_WaitForOrder>
{
public:
    L_WaitForOrder();

private:
    float timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};