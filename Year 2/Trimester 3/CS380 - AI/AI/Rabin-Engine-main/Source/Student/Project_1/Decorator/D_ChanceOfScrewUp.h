#pragma once
#include "BehaviorNode.h"

class D_ChanceOfScrewUp : public BaseNode<D_ChanceOfScrewUp>
{
public:
    D_ChanceOfScrewUp();

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};