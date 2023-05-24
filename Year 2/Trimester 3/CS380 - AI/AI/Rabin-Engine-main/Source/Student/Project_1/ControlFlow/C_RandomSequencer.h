#pragma once
#include "BehaviorNode.h"

class C_RandomSequencer : public BaseNode<C_RandomSequencer>
{
public:
    C_RandomSequencer(void);

private:
    void ChooseRandomNode(void);

    size_t currentIndex{};
    std::vector<bool> chosenNodes{};

protected:
    virtual void on_enter() override;
    virtual void on_update(float dt) override;
};