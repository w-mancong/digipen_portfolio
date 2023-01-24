/*!*****************************************************************************
\file cipher.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 3
\date 10-09-2022
\brief
This file contain function definition to encode, decode, read bits and print bits
*******************************************************************************/
#include <iostream>
#include <cstdint>
#include "cipher.h"

/*!*****************************************************************************
    \brief To print out to console based on the value of the bit at the specific position

    \param [in] v: Single byte containing the encrypted data
    \param [in] pos: Bit position to be checked with and printed out
*******************************************************************************/
void print_bit(char v, int32_t pos) 
{
  char one = 1;
  if ( v & one << pos ) { std::cout << "1"; } else { std::cout << "0"; }
}

/*!*****************************************************************************
    \brief Print the bits that was encoded

    \param [in] buffer: Buffer containing the encrypted text
    \param [in] start_pos: Starting index
    \param [in] how_many: Total number of bits used when encrypting the text
*******************************************************************************/
void print_bits(char* buffer, int32_t start_pos, int32_t how_many) 
{
  std::cout << "Bits: ";
  for (int32_t i{}; i < how_many;) // for each char
  { 
    char *ch = buffer + (start_pos + i) / 8;
    for (int32_t j{}; j < 8 && i < how_many; ++j, ++i)
    { // from more significant to less
      print_bit( *ch, j );
    }
  }
  std::cout << '\n';
}

/*!*****************************************************************************
    \brief Determine the bits at the specified location

    \param [in] buffer: Encrypted text to have it's bit read
    \param [in] i: Position of the bit to be checked

    \return 1 if that bit position is 1, else 0
    7 6 5 4 3 2 1 0  <- Bit position
    - - - - 0 1 1 0

    if i = 2, function will return 1 as Bit position indexed 2 is 1
*******************************************************************************/
int32_t read_bit(char const* buffer, int32_t i) 
{
  char const *ch = buffer + i / 8;
  int32_t pos = i % 8;
  return (*ch & 1 << pos) ? 1 : 0;
}

/*!*****************************************************************************
    \brief To decrypt encoded text into it's plain text

    \param [in] ciphertext: Encrypted text to be turn into plain text
    \param [in] num_chars: Total number of characters in the original plain text
    \param [out] plaintext: Storing the decrypted text into this string
*******************************************************************************/
void decode(char const* ciphertext, int32_t num_chars, char* plaintext) 
{
  int32_t pos{};
  for(int32_t i{}; i < num_chars; ++i ) 
  {
    // read 2 bits for group (00,01,10,11)
    int32_t group_index = read_bit(ciphertext, pos) + 2 * read_bit(ciphertext, pos + 1);
    int32_t index{}; // index inside group
    pos += 2;

    for (int32_t j{}; j < group_index + 1; ++j)
    {
      index += (read_bit(ciphertext, pos) << j);
      ++pos;
    }
    plaintext[i] = 'a' + ((1 << (group_index + 1)) - 2) + index;
  }
  plaintext[num_chars] = 0; // null terminate final result
}

/*!*****************************************************************************
    \brief To encrypt plain text into into each byte by bit twiddling the data

    \param [in] plaintext: Text to be encrypted
    \param [out] encryptedtext: Encrypted text will be stored into this string
    \param [out] num_bits_used: Total number of bytes written into the encrypted text
*******************************************************************************/
void encode(char const * plaintext, char* encryptedtext, int32_t *num_bits_used ) 
{
  *num_bits_used = 0;
  // to get the binary code based on the character
  auto get_binary_code = [](char ch, int32_t &loop)
  {
    uint8_t binary_code { 0 };
    switch (ch)
    {
      case 'a': case 'b':
      {
        binary_code =     (static_cast<int8_t>(ch) - static_cast<int8_t>('a')) * 4;
        loop = 3;
        break;
      }
      case 'c': case 'd': case 'e': case 'f':
      {
        binary_code = 1 + (static_cast<int8_t>(ch) - static_cast<int8_t>('c')) * 4;
        loop = 4;
        break;
      }
      case 'g': case 'h': case 'i': case 'j':
      case 'k': case 'l': case 'm': case 'n':
      {
        binary_code = 2 + (static_cast<int8_t>(ch) - static_cast<int8_t>('g')) * 4;
        loop = 5;
        break;
      }
      case 'o': case 'p': case 'q': case 'r':
      case 's': case 't': case 'u': case 'v':
      case 'w': case 'x': case 'y': case 'z':
      {
        binary_code = 3 + (static_cast<int8_t>(ch) - static_cast<int8_t>('o')) * 4;
        loop = 6;
        break;
      }
    }
    return binary_code;
  };

  int32_t bit_pos{ 0 }, index{ 0 };
  char const *text = plaintext;
  uint8_t bits{ 0 };

  // to reset bit position and use the next 8 bits to store encrypted text
  auto reset = [&](void)
  {
    if (bit_pos >= 8)
    {
      bit_pos = 0, bits = 0;
      ++index;
    }
  };

  while (*text)
  {
    int32_t loop { 0 };
    uint8_t binary_code = get_binary_code(*text, loop);

    for (int32_t i{}; i < loop; ++i)
    {
      bits |= (binary_code & (1 << i) ? 1 : 0) << bit_pos++;
      *(encryptedtext + index) = static_cast<char>(bits);
      ++*num_bits_used;
      reset();
    }
    ++text;
  }
}
