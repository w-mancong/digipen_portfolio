#pragma once
#include "BehaviorNode.h"

class L_OrderFood : public BaseNode<L_OrderFood>
{
public:
    L_OrderFood();

private:
    float angle{}, flag{};
    bool leftQueue{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
    virtual void on_exit() override;
};