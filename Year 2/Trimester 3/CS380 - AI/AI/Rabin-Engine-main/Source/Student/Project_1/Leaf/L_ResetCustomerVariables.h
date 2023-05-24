#pragma once
#include "BehaviorNode.h"

class L_ResetCustomerVariables : public BaseNode<L_ResetCustomerVariables>
{
public:
    L_ResetCustomerVariables();

protected:
    virtual void on_enter() override;
};