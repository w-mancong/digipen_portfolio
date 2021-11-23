/*!
@file       q.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 5
@date       07/10/2021
@brief      Helps buyer decide which sleeping bag to buy based on weather
*//*__________________________________________________________________________*/

#include <stdio.h>

/*!
@brief      Based on weather and humidity, helps the buyer decide which
			sleeping bag type they should buy
@param		temperature: in degrees celsuis
			humidity: level
*//*__________________________________________________________________________*/
void sleeping_bag(signed char temperature, unsigned char humidity)
{		
	if(-50 > temperature || 100 < temperature || 100 <= humidity)
	{
		printf("Invalid input!\n");
		return;
	}
	
	const char* bag[3] = { "Summer", "3-Season", "Winter" };
	int bagType = 0;

	if(15 <= temperature && 30 > temperature)
		bagType = 1;
	else if(15 > temperature)
		bagType = 2;
					
	const char* insulation[2] = { "Synthetic", "Down" };
	int insu = 0;
	
	if (40 >= humidity)
		insu = 1;
	
	printf("The temperature is %d*C, humidity is %d%%.\n", temperature, humidity);
	printf("The best sleeping bag is %s type insulated with %s.\n", bag[bagType], insulation[insu]);
}
