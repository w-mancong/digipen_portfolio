#pragma once
#include "BehaviorNode.h"

class D_RunOnlyOnce : public BaseNode<D_RunOnlyOnce>
{
public:
    D_RunOnlyOnce();

private:
    bool ran{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};