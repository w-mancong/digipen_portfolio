double poly_opt(double a[],double x, long degree)
{
    long int i = degree;
    double r1 = 0.0, r2 = 0.0, r3 = 0.0, r4 = 0.0;

    double const x2 = x * x;
    double const x3 = x * (x * x);
    double const x4 = (x * x) * (x * x);
    double const x5 = (x * x) * (x * (x * x));
    double const x6 = ((x * x) * x) * (x * (x * x));
    double const x7 = (x * x) * ((x * x) * (x * (x * x)));
    double const x8 = ((x * x) * (x * x)) * ((x * x) * (x * x));

    for (; i >= 7; i -= 8)
    {
        r1 = x8 * r1 +
             a[i - 0] * x7 +
             a[i - 2] * x5;

        r2 = x8 * r2 +
             a[i - 4] * x3 +
             a[i - 6] * x;

        r3 = x8 * r3 +
             a[i - 1] * x6 +
             a[i - 3] * x4;

        r4 = x8 * r4 +
             a[i - 5] * x2 +
             a[i - 7];
    }

    r1 += (r2 + (r3 + r4));

    for (; i >= 0; --i)
        r1 = a[i] + x * r1;

    return r1;
}