#pragma once
#include "BehaviorNode.h"

class L_TakeOrderFromCustomer : public BaseNode<L_TakeOrderFromCustomer>
{
public:
    L_TakeOrderFromCustomer();

private:
    float angle{}, flag{};
    float timer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};