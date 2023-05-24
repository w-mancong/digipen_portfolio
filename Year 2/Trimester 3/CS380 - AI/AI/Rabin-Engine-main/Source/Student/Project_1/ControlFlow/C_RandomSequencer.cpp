#include <pch.h>
#include "C_RandomSequencer.h"

C_RandomSequencer::C_RandomSequencer() {}

void C_RandomSequencer::ChooseRandomNode(void)
{
    // loop until a chosenNode is not used
    while (true)
    {
        currentIndex = RNG::range(0ULL, children.size() - 1ULL);
        if (chosenNodes[currentIndex])
            continue;
        chosenNodes[currentIndex] = true;
        break;
    }
}

void C_RandomSequencer::on_enter()
{
    BehaviorNode::on_enter();
    if (chosenNodes.size())
        return;
    chosenNodes.resize(children.size());
    std::fill(chosenNodes.begin(), chosenNodes.end(), false);
}

void C_RandomSequencer::on_update(float dt)
{
    // choose random node to start
    ChooseRandomNode();
    BehaviorNode* currentNode = children[currentIndex];
    currentNode->tick(dt);

    // if any child fails, the node fails
    if (currentNode->failed())
        on_failure();
    // if all children succeed, the node succeeds
    else if (currentNode->succeeded())
    {
        // check the number of true in the vector
        if (std::count_if(chosenNodes.begin(), chosenNodes.end(), [](bool i)
            {
                return i;
            }) == chosenNodes.size())
        {
            on_success();
            std::fill(chosenNodes.begin(), chosenNodes.end(), false);
        }
    }
}
