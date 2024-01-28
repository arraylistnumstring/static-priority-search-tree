#include <cstdlib>
#include <ctime>
#include <iostream>

#include "template-class.h"

int main()
{
	srand(time(nullptr));

	TemplateClass<int> *pt_arr = new TemplateClass<int>[10];
	for (int i = 0; i < 10; i++)
	{
		pt_arr[i].dim1_val = rand();
		pt_arr[i].dim2_val = rand();
	}

	for (int i = 0; i < 10; i++)
		std::cout << pt_arr[i].dim1_val << ", " << pt_arr[i].dim2_val << '\n';

	return 0;
}
