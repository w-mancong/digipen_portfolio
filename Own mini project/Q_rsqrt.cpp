float Q_rsqrt(float number)
{
	float x2 = number * 0.5f, y = number;
	float const threeHalfs = 1.5f;

	long i = *(long*)&y;					// Reserving the bits of float and storing it into an intergal value so that can do bit manipulation
	i = 0x5f3759df - (i >> 1);				// 0x5f3759df is some number of Mantissa and Exponent with an offset, - (i >> 1) is to make the exponent be inverse
	y = *(float*)&i;						// Converting back to floating number
    // for (size_t i = 0; i < 3; ++i)
        y *= threeHalfs - x2 * y * y;		// Newton's method	(can decide over accuracy/speed) 1% off
	return y;
}