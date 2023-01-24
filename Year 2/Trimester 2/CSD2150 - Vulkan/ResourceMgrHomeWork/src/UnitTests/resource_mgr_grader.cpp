#include "../resource_mgr.h"
#include "TestGuids.h"
#include "TestResourceMgr.h"

int main(void)
{
    float TotalScore = 0;
    TotalScore += resource::unitest::guids::Evaluate() * 0.25f;
    TotalScore += resource::unitest::resource_type_registration::Evaluate() * 0.75f;

    printf( "\n\nFinal Score = %3.0f%%", TotalScore * 100 );

    return 0;
}