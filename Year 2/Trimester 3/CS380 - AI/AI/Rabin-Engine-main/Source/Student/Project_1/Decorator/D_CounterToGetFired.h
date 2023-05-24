#pragma once
#include "BehaviorNode.h"

class D_CounterToGetFired : public BaseNode<D_CounterToGetFired>
{
public:
    D_CounterToGetFired();

private:
    int counter = 0;

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};