#pragma once
#include "BehaviorNode.h"

class L_EmptyTable : public BaseNode<L_EmptyTable>
{
public:
    L_EmptyTable();

protected:
    virtual void on_enter() override;
};