#pragma once
#include "BehaviorNode.h"

class D_ChanceToMessUpDish : public BaseNode<D_ChanceToMessUpDish>
{
public:
    D_ChanceToMessUpDish();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};