#pragma once
#include "BehaviorNode.h"

class L_AngryCustomer : public BaseNode<L_AngryCustomer>
{
public:
    L_AngryCustomer();

protected:
    virtual void on_enter() override;
};