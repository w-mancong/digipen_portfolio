#include <stdio.h>

void sq_root_compute_varargs(unsigned int n1, ...);
void sq_root_compute_array(int num_of_elements, unsigned int *array_of_elements);



void test1()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 0);


}

void test2()
{
		printf("sq_root_compute_array \n");

		unsigned int a[1] = {3};

		sq_root_compute_array(1, a );
}


void test3()
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

}	

void test4()
{

		printf("sq_root_compute_array \n");

		unsigned int a[8] = {3,25, 169,9830, 1169, 19810, 1369, 99810};

		sq_root_compute_array(8, a );
}	

void test5()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[1] = {3};

		sq_root_compute_array(1, a );
}
	

void test6()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 9810, 
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[2] = {3,9810};

		sq_root_compute_array(2, a );
}	

void test7()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 9810, 
								(unsigned int) 99810, 
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[3] = {3,9810, 99810};

		sq_root_compute_array(3, a );
}	

void test8()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 9810, 
								(unsigned int) 99810,
								(unsigned int) 19810,
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[4] = {3,9810, 99810,19810};

		sq_root_compute_array(4, a );
		
}	

void test9()
{
		printf("sq_root_compute_varargs \n");

		sq_root_compute_varargs((unsigned int) 3,
								(unsigned int) 9810, 
								(unsigned int) 99810,
								(unsigned int) 19810, 
								(unsigned int) 169, 
								(unsigned int) 0);


		printf("sq_root_compute_array \n");

		unsigned int a[5] = {3,9810, 99810,19810, 169};

		sq_root_compute_array(5, a );
		
}	

void test10()
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

void test(int i)
{
    switch(i)
    {
        case 1:
        test1();
        break;
        case 2:
        test2();
        break;
        case 3:
        test3();
        break;
        case 4:
        test4();
        break;
        case 5:
        test5();
        break;
        case 6:
        test6();
        break;
        case 7:
        test7();
        break;
        case 8:
        test8();
        break;
        case 9:
        test9();
        break;
        case 10:
        test10();
        break;

        default:
        test1();
    }
}

