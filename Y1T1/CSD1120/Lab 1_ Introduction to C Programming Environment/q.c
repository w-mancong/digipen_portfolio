// @author: Wong Man Cong (w.mancong@digipen.edu)
#include <stdio.h>

int exponent(int x, int y);

int main(void)
{
    printf("Enter base and power: ");
    int base, power;
    scanf("%d %d", &base, &power);
    int result = exponent(base, power);
    printf("%d^%d is: %d\n", base, power, result);
    return 0;
}

int exponent(int x, int y)
{
    int i = x, j = 1;
    while (j < y)
    {
        i *= x;
        ++j;
    }
    return i;
}
