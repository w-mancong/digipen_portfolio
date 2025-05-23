#include <iostream>
#include <climits>
#include <cstdint>
#include <sstream>
#include <string>
#include <array>
#include <exception>

#ifndef USE_STL_BITSET
  #include "bitset.h"
	namespace ns = HLP3;
#else
  #include <bitset>
	namespace ns = std;
#endif

// declarations ...
namespace {
  void test1();  void test2();  void test3();  void test4();  void test5();
  void test6();  void test7();  void test8();  void test9();  void test10();
  void test11(); void test12(); void test13();
#ifndef USE_STL_BITSET
  void test14(); void test15(); void test16(); void test17();
  void test18(); void test19(); void test20(); void test21();
#endif
  void run(std::size_t& testIndex, void (*test)());
}

int main() {
#ifndef USE_STL_BITSET
  constexpr uint32_t max_tests{21};
#else
  constexpr uint32_t max_tests{13};
#endif
  std::array<void (*)(), max_tests> tests = { 
		test1, test2, test3,  test4,  test5,  test6, test7,
		test8, test9, test10, test11, test12, test13,
#ifndef USE_STL_BITSET
		test14, test15, test16, test17,
		test18, test19, test20, test21,
#endif
	};

	std::size_t test_index{};
	for (std::size_t i{}; i < tests.size(); ++i) {
		run(test_index, tests[i]);
	}
}

namespace {
	// Note: size of std::bitset<N> classes may be implementation defined,
	// but your implementation must have the smallest size possible: size_t(void*)

void test1() {
  constexpr size_t INDEX = 0;
  ns::bitset<16> bitset;

  bitset.set(INDEX, true);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been set."};
  }
  bitset.set(INDEX, false);
  bitset.set(INDEX + 1, true);
  if (bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been cleared."};
  }
}

void test2() {
  constexpr size_t INDEX = 15;
  ns::bitset<INDEX + 1> bitset;

  bitset.set(INDEX, true);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been set."};
  }
  bitset.set(INDEX, false);
  bitset.set(INDEX - 1, true);
  if (bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been cleared."};
  }
}

void test3() {
  constexpr size_t INDEX = 0;
  ns::bitset<16> bitset;

  bitset.set(INDEX, true);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been set."};
  }
  bitset.reset(INDEX);
  bitset.set(INDEX + 1, true);
  if (bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been cleared."};
  }
}

void test4() {
  constexpr size_t INDEX = 15;
  ns::bitset<INDEX + 1> bitset;

  bitset.set(INDEX, true);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit should have been set."};
  }
  bitset.flip(INDEX);
  bitset.set(INDEX - 1, true);
  if (bitset.test(INDEX)) {
    throw std::runtime_error{"Bit should have been cleared."};
  }
}

void test5() {
  constexpr size_t INDEX = 15;
  ns::bitset<INDEX + 1> bitset;

  bitset.reset(INDEX);
  if (bitset.test(INDEX)) {
    throw std::runtime_error{"Bit should have been cleared."};
  }
  bitset.flip(INDEX);
  bitset.set(INDEX - 1, false);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit should have been set."};
  }
}

void test6() {
  ns::bitset<CHAR_BIT> bitset;
  try {
    (void)bitset.test(bitset.size());
    throw std::runtime_error {
      "Test of a bit outside of range should have resulted in "
      "an exception."
    };
  } catch (const std::out_of_range&) {
    // All good; swallow the exception and continue as normal.
  }
}

void test7() {
  constexpr size_t INDEX = 15;
  ns::bitset<INDEX + 1> bitset;

  bitset.set(INDEX, true);
  if (!bitset.test(INDEX)) {
    throw std::runtime_error{"Bit has not been set."};
  }
  bitset.set(INDEX, false);
  if (bitset[INDEX]) {
    throw std::runtime_error{"Bit has not been cleared."};
  }
}

void test8() {
  ns::bitset<5> bitset;
  bitset.set(1);
  bitset.set(2);

  const std::string s = bitset.to_string();
  if (s != "00110") {
    std::cout << s << std::endl;
    throw std::runtime_error{"Converstion to string did not work."};
  }
}

void test9() {
  ns::bitset<5> bitset;
  bitset.set(1);
  bitset.set(2);

  const std::string s = bitset.to_string('_');
  if (s != "__11_") {
    std::cout << s << std::endl;
    throw std::runtime_error{"Converstion to string did not work."};
  }
}

void test10() {
  ns::bitset<5> bitset;
  bitset.set(1);
  bitset.set(2);

  const std::string s = bitset.to_string('F', 'T');
  if (s != "FFTTF") {
    std::cout << s << std::endl;
    throw std::runtime_error{"Converstion to string did not work."};
  }
}

void test11() {
  ns::bitset<5> bitset;
  bitset.set(1);
  bitset.set(2);
  if (bitset.count() != 2) {
    throw std::runtime_error{"Counting set bits did not work."};
  }
}

void test12() {
  ns::bitset<5> bitset;
  bitset.reset(1);
  bitset.set(2);
  if (bitset.count() != 1) {
    throw std::runtime_error{"Counting set bits did not work."};
  }
}

void test13() {
  if constexpr (sizeof(ns::bitset<CHAR_BIT * 8>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

#ifndef USE_STL_BITSET

void test14() {
  if constexpr (sizeof(ns::bitset<1>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test15() {
  if constexpr (sizeof(ns::bitset<CHAR_BIT>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test16() {
  if constexpr (sizeof(ns::bitset<CHAR_BIT + 1>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test17() {
  if constexpr (sizeof(ns::bitset<CHAR_BIT * 2>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test18() {
  if constexpr (sizeof(ns::bitset<CHAR_BIT * 2 + 1>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test19() {
  constexpr size_t LARGEST_INT_SIZE = sizeof(long long) / CHAR_BIT;
  if constexpr (sizeof(ns::bitset<LARGEST_INT_SIZE * CHAR_BIT + 1>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test20() {
  constexpr size_t LARGE_VALUE = 1024 * 1024;
  if constexpr (sizeof(ns::bitset<LARGE_VALUE * CHAR_BIT + 1>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

void test21() {
  constexpr size_t LARGE_VALUE = 1024 * 1024;
  if constexpr (sizeof(ns::bitset<LARGE_VALUE * CHAR_BIT>) != 8) {
    throw std::runtime_error {
      "Required size of the object is exactly 8 bytes.\n"
      "Data must be stored in a dynamic memory."
    };
  }
}

#endif

void run(std::size_t& testIndex, void (*test)()) {
  ++testIndex;
  try {
    test();
    std::cout << "Test " << testIndex << " completed successfully." << std::endl;
  } catch (const std::string& exception) {
    std::cout << "Test " << testIndex << " failed. " << exception << std::endl;
  } catch (const std::exception& exception) {
    std::cout << "Test " << testIndex << " failed. " << exception.what() << std::endl;
  } catch (...) {
    std::cout << "Test " << testIndex << " failed due to an unexpected exception!" << std::endl;
  }
}
} // end anonymous namespace
