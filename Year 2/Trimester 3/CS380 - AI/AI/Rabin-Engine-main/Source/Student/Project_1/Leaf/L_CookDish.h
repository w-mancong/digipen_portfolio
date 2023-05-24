#pragma once
#include "BehaviorNode.h"

class L_CookDish : public BaseNode<L_CookDish>
{
public:
    L_CookDish();

private:
    float timer{}, angle{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};