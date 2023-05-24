#pragma once
#include "BehaviorNode.h"

class D_RunTillSuccess : public BaseNode<D_RunTillSuccess>
{
public:
    D_RunTillSuccess();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};