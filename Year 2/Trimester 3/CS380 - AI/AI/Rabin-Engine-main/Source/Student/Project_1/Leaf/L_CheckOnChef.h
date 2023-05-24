#pragma once
#include "BehaviorNode.h"

class L_CheckOnChef : public BaseNode<L_CheckOnChef>
{
public:
    L_CheckOnChef();

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