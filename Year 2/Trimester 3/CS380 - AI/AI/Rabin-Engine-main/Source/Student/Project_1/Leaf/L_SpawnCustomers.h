#pragma once
#include "BehaviorNode.h"

class L_SpawnCustomers : public BaseNode<L_SpawnCustomers>
{
public:
    L_SpawnCustomers();

protected:
    virtual void on_enter() override;
};