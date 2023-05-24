#pragma once
#include "BehaviorNode.h"

class L_HaveItToGo : public BaseNode<L_HaveItToGo>
{
public:
    L_HaveItToGo();

protected:
    float timer;

    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};