#include <iostream>

float MultiplyWith141(short in)
{
    if (!in)
        return 0.0f;

    union Result
    {
        int final_value{ 0 };
        float value;
    } to_return_float;

    // 0b101101 is 1.40625 in binary based on IEEE754
    auto get_bits = [&](size_t pos)
    {
        return 0b101101 & (0b1 << pos);
    };

    auto get_msb = [&](int n)
    {
        if (!in)
            return 0;

        int msb = 0;
        n = std::abs(n);
        while ((n >>= 1))
            ++msb;
        return msb;
    };

    // initial exponent value
    unsigned exponent{ static_cast<unsigned>(get_msb(in)) };
    int res{};
    // looping thru the constant bits of 1.40625 = 0b101101
    for (size_t i{}; i < 6; ++i)
    {
        if (!get_bits(i))
            continue;
        res += static_cast<int>(in << i);
    }

    // Absoluting this value becuz from now on I'm just wanna work with it's mantissa bits
    res = std::abs(res);
    /*
      after multiplying, need to find new exponent value
      using magic number 5 becuz the most signficant bit of 1.40625 binary form is 5
    */
    if ((res >> (exponent + 5)) > 1)
    {
        short shift = 5 + (exponent += 1);
        while ((res >> shift) > 1)
            shift = (++exponent) + 5;
    }

    /*
      There is two cases here, final mantissa value is <= 23
      OR value is > 23
      if final mantissa value is <= 23, can just assign all the bits
      else only assign 23 bits, if last bit is 1 then add 1 to the value of the 23 bits
    */
    unsigned mantissa{}, res_msb{ static_cast<unsigned>(get_msb(res)) };
    mantissa |= res;
    if (23 < res_msb)
    {
        size_t const shift = sizeof(res) * 8 - 23; // value to be shifted
        // Only take the first 23 bits of value as mantissa
        mantissa |= (res >> shift);
        mantissa += (res & (0b1 << shift)) ? 1 : 0;
    }

    // remove the most significant bit becuz I'm only using the back as mantissa value
    mantissa &= ~(0b1 << res_msb);
    /*
      1) assign sign bit
      2) assign exponents
      3) assign mantissa
    */
    exponent += 127;
    if (in < 0)
        to_return_float.final_value |= 0b1 << 31;
    to_return_float.final_value |= exponent << 23;
    to_return_float.final_value |= mantissa << (23 - res_msb);

    return to_return_float.value;
}

/*
	23 zeros; 00000000000000000000000


			  101100111111111010011
	0 10001110 01100111111111010011010 (46078.6)
	0 10001111 01100111111111010011010 (92157.2)
	         1 01100111111111010011
	32767
*/
int main(void)
{
	short constexpr const test{ std::numeric_limits<short>::max() };
	//short constexpr const test{ 20 };

	int res = test + (test >> 2) + (test >> 3) + (test >> 5);
	std::cout << res << std::endl;

	//double test{ 80.0f };
	std::cout << MultiplyWith141(static_cast<short>(test)) << std::endl;
	std::cout << test * 1.40625 << std::endl;
}