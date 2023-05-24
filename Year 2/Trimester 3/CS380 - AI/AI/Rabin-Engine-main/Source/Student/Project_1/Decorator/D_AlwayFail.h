#pragma once
#include "BehaviorNode.h"

class D_AlwayFail : public BaseNode<D_AlwayFail>
{
public:
    D_AlwayFail();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};