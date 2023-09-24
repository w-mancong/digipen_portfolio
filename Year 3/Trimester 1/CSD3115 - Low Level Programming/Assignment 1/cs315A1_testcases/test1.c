#include <stdio.h>


void sq_root_compute_varargs(unsigned int n1, ...);
void sq_root_compute_array(int num_of_elements, unsigned int *array_of_elements);

int main()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 25,
								(unsigned int) 169,
								(unsigned int) 9810, 
								(unsigned int) 1169,
								(unsigned int) 19810, 
								(unsigned int) 1369,
								(unsigned int) 99810, 
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[8] = {3,25, 169,9830, 1169, 19810, 1369, 99810};

		sq_root_compute_array(8, a );

}
	