
typedef struct L_square_struct
{
    double A, B, C, D;
    int count;
} L_square_struct;

void update( L_square_struct *, double x, double y);
double get_current_slope(L_square_struct *);
void init_l_square_struct(L_square_struct *l);