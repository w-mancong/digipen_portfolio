#pragma once
#include "BehaviorNode.h"

class L_CheckOnCashier : public BaseNode<L_CheckOnCashier>
{
public:
    L_CheckOnCashier();

private:
    float timer{}, angle{}, flag{};
    int counter{};
    Vec3 targetPosition{};
    bool interactWithCashier, doneInteraction;

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};