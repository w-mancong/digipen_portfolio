/*!
@file       gamelib.c
@author     Wong Man Cong (w.mancong@digipen.edu)
@course     CSD 1120
@section    B
@assignment 9
@date       12/11/2021
@brief      Accordion solitaire game for one player
*//*__________________________________________________________________________*/

#include <stdio.h>
#include "gamelib.h"

/*!
@brief	fill in the cards from the right to the empty space starting from
		position

@param  game: reference to the array with the deck of cards
		position: position of the card that have been placed over another card
*//*__________________________________________________________________________*/
void del_card(Card game[], CardIndex position)
{
	while(game[position].suit != '0' && game[position].rank != '0')
	{
		game[position].suit = game[position + 1].suit;
		game[position].rank = game[position + 1].rank;
		++position;
	}
}

/*!
@brief	load and store the values into 'game' array

@param  str: loaded deck of cards
		game: array to store the deck of cards
*//*__________________________________________________________________________*/
void load_game(const char str[], Card game[])
{
	// set all card suit's and rank to null terminator
	for (int i = 0; i < N; ++i)
	{
		game[i].rank = '\0';
		game[i].suit = '\0';
	}

	int suit_index = 0, rank_index = 1, card_index = 0, check_index = 0;
	while (game[check_index].suit != '0' && game[check_index].rank != '0')
	{
		game[card_index].suit = str[suit_index];
		game[card_index++].rank = str[rank_index];
		suit_index += 3;
		rank_index += 3;
		check_index = card_index - 1;
	}
}

/*!
@brief	if a 3 position move is possible, take it. else make a 1 position move

@param  game: array to the deck of cards
*//*__________________________________________________________________________*/
void play_game(Card game[])
{
	int card_index = 1;
	while (1)
	{
		if (game[card_index].suit == '0' && game[card_index].rank == '0')
			break;
		if (card_index - 3 >= 0)
		{
			if (game[card_index - 3].suit == game[card_index].suit || game[card_index - 3].rank == game[card_index].rank)
			{
				game[card_index - 3].suit = game[card_index].suit;
				game[card_index - 3].rank = game[card_index].rank;
				del_card(game, (unsigned char)card_index);
				card_index = 0;
			}
		}
		if (game[card_index - 1].suit == game[card_index].suit || game[card_index - 1].rank == game[card_index].rank)
		{
			game[card_index - 1].suit = game[card_index].suit;
			game[card_index - 1].rank = game[card_index].rank;
			del_card(game, (unsigned char)card_index);
			card_index = 0;
		}
		++card_index;
	}
}

/*!
@brief	display the final arrangement of cards after the play

@param  game: array to the deck of cards
*//*__________________________________________________________________________*/
void display_game(const Card game[])
{
	int card_index = 0;
	do
	{
		printf("%c%c ", game[card_index].suit, game[card_index].rank);
	} while (game[card_index].suit != '0' && game[card_index++].suit != '0');
	printf("\n");
}
