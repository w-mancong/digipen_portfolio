#pragma once
#include "BehaviorNode.h"

class L_AngryCook : public BaseNode<L_AngryCook>
{
public:
    L_AngryCook();

private:
    float timer{}, angle{}, flag{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};