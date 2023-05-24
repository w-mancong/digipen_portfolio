#pragma once
#include "BehaviorNode.h"

class L_ChangeAgentColour : public BaseNode<L_ChangeAgentColour>
{
protected:
    virtual void on_enter() override;
};