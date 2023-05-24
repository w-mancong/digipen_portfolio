#pragma once
#include "BehaviorNode.h"

class D_AlwaySucceed : public BaseNode<D_AlwaySucceed>
{
public:
    D_AlwaySucceed();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};