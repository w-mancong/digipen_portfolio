#pragma once
#include "BehaviorNode.h"

class L_GetsFired : public BaseNode<L_GetsFired>
{
public:
    L_GetsFired();

private:
    float angle{}, scale{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};