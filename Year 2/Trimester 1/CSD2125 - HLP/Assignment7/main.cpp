#include "index_sequence.h"

#ifdef CPP11
	#include "cpp11.h"
#elif CPP17
	#include "cpp17.h"
#elif TMP11
	#include "tmp11.h"
#elif TMP17
	#include "tmp17.h"
#endif

int main() {
	make_sequence<N>::print();
}
