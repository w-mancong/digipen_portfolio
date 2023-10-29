#include "compute_least_square.h"

void init_l_square_struct(L_square_struct *l)
{
    l->A=l->B=l->C=l->D=0;
    l->count=0;
}

void update(L_square_struct *l, double x, double y)
{
    if(x!=0)
    {
		l->count++;
		l->A += x*x;
		l->B += x;
		l->C += x*y;
		l->D += y;
    }
}

double get_current_slope(L_square_struct *l)
{
    return (l->count * l->C - l->B * l->D) / 
	(l->count * l->A - l->B * l->B);
}
