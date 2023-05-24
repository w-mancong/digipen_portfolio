#pragma once
#include "BehaviorNode.h"

class L_EatAtTable : public BaseNode<L_EatAtTable>
{
public:
    L_EatAtTable();

private:
    float angle{}, flag{};
    float eatTimer{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};