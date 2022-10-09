#include "readline.h"
#include <stdio.h>  // fopen/fclose, printf
#include <stdlib.h> // malloc/free

void simple_test( char const * );
void read_long_line( int , char , int , int );
void test0( void );
void test1( void );
void test2( void );
void test3( void );
void test4( void );
void test5( void );
void test6( void );
void test7( void );

// read first line from file and dump it
void simple_test( char const * filename ) {
  FILE * file_handle = fopen( filename, "r" );
  if ( ! file_handle ) {
    printf( "Cannot open file for reading\n" );
  }

  char * str = readline( file_handle );
  printf( "\n----------------------\n" );
  printf( "-->%s<--", str );
  printf( "\n----------------------\n" );
  fclose( file_handle );
  free( str );
}

/*
creates a file with size occurrences of character ch.
If newline_in_middle is true, a newline is added followed by size occurrences
of character ch+1.
If newline_in_end is set, newline is added at end of the file.

Next, readline() is called and returned value compared to the expected:
ch size times followed by null-character (ASCII 0)
*/
void read_long_line( int size, char ch, int newline_in_middle, int newline_in_end ) {
  // create a file
  char const * filename = "temp_file";
  FILE * file_handle = fopen( filename, "w" ); // write text
  if ( ! file_handle ) {
    printf( "Cannot open file for reading\n" );
    return;
  }

  int total_size = size + (newline_in_middle?1+size:0) + (newline_in_end?1:0);
  char *buffer = (char*) malloc( total_size );
  if ( buffer ) {
    int i;
    for ( i=0; i<size; ++i ) {
      buffer[i] = ch;
    }
    if ( newline_in_middle ) {
      buffer[ i ] = '\n'; // position = size
      ++i;
      for ( ; i<2*size+1; ++i ) {
        buffer[i] = ch+1;
      }
    }
    if ( newline_in_end ) {
      buffer[ i ] = '\n'; 
    }
  }

  fwrite( buffer, total_size, 1, file_handle );
  fclose( file_handle );

  // creation of file is done - now open the same file and pass to readline
  file_handle = fopen( filename, "r" );
  if ( ! file_handle ) {
    printf( "Cannot open file for reading\n" );
    return;
  }

  char *str = readline( file_handle );
  int i;
  for ( i=0; i<size; ++i ) {
    if ( str[i] != ch ) {
      printf( "error: str[%i] = %c, when %c was expected\n", i, str[i], ch );
    }
  }
  if ( str[i] != 0 ) {
    printf( "error: str[%i] = %c, when null-char was expected\n", i, str[i] );
  }
  fclose( file_handle );
  remove( filename );
  free( str );
  free( buffer );
}

void test0( void ) { simple_test( "in0" ); } /* 3 chars, newline in the end */
void test1( void ) { simple_test( "in1" ); } /* 7 chars, newline in the end */
void test2( void ) { simple_test( "in2" ); } /* 3 chars, newline, 3 chars, new line in the end */
void test3( void ) { simple_test( "in3" ); } /* 7 chars, newline, 7 chars, new line in the end */
void test4( void ) { simple_test( "in4" ); } /* 3 chars, NO new line in the end */
void test5( void ) { simple_test( "in5" ); } /* empty file, but new line in the end. Filesize = 1 */
void test6( void ) { read_long_line( 1<<12, '.', 1, 1 ); }
void test7( void ) { read_long_line( 1<<20, '.', 0, 1 ); }
void test8( void ) { read_long_line( 1<<22, '.', 1, 1 ); }
void test9( void ) { read_long_line( 1<<26, '.', 0, 0 ); } /* about 0.5 sec */

void (*pTests[])( void ) = {
  test0,test1,test2,test3,test4,
  test5,test6,test7,test8,test9,
};

int main( int argc, char ** argv )  {
  if ( argc == 2 ) {
    int test = 0;
    sscanf(argv[1],"%i",&test);
    pTests[test]();
  }
}
