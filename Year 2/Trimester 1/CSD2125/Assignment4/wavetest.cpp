/* wavetest.cpp
** -- simple driver for WAVE file functions
** cs225 5/25/2021
*/
// in: 538, out: 644
#include <iostream>
#include <cmath>
#include "wave.h"

// constexpr means constant expression that can be evaluated
// at compile-time and it is possible then for the compiler
// to optimize code by removing references to the variable
// TWOPI in any expressions with the value associated with
// that variable.
// Here, we make the constant expression a read-only variable too
constexpr double const TWOPI {6.283185307};

// ring modulator filter function ...
// ring modulator filter data
struct RingModulatorData 
{
  double argument;
  double argument_inc;
  float mix;
};
// initialize ring modulator filter data
void ringModulatorInit(struct RingModulatorData *rmd, float f, float R, float m);
// ring modulator filter function
short ringModulator(short v, void *vp);

// driver ...
int main(int argc, char *argv[]) {
  // need at least two command line arguments:
  // the first argument is the name of the program
  // the second is the name of the WAVE audio data file
  if (argc != 2) {
    std::cout << "*** name of a 16-bit WAVE file is expected ***\n";
    return 0;
  }

  // open file, and dump its contents into char buffer file_contents
  char *file_contents = waveRead(argv[1]);
  // parse the WAVE file
  // special note: make sure to check if file_contents is nullptr or not
  // special note: make sure to delete the memory pointed to by file_contents!!!
  WaveData wave_data = waveParse(file_contents);
  if (!wave_data) {
    std::cout <<  "*** not a valid 16-bit WAVE file ***\n";
    return -1;
  }
  
  // filter channel 0 with ring modulator
  struct RingModulatorData rmd;
  ringModulatorInit(&rmd, 350.f, static_cast<float>(waveSamplingRate(wave_data)), 0.5f);
  waveSetFilterData(wave_data, &rmd);
  waveFilter(wave_data, 0, ringModulator);

  // scale channel 0 by a factor of 1.2 (with clipping)
  // don't make assumption about scale factor - the autograder
  // will use a variety of different values
  unsigned short bd = waveBoostData(1.2f);
  waveSetFilterData(wave_data, &bd);
  waveFilter(wave_data, 0, waveBoost);

  // scale channel 0 by a factor of 0.5
  waveFilter(wave_data, 0, waveCut6DB);

  // if present, scale channel 1 by a factor of 1.41 (with clipping)
  if (waveChannelCount(wave_data) == 2) {
    waveFilter(wave_data, 1, waveBoost3DB);
  }

  waveWrite(wave_data, "WaveTest.wav");
  waveDestroy(wave_data);
}

// initialize ring modulator filter data
void ringModulatorInit(struct RingModulatorData *rmd, float f, float R, float m) {
  rmd->argument = 0.0;
  rmd->argument_inc = TWOPI * f / R;
  rmd->mix = m;
}

// ring modulator filter function
short ringModulator(short v, void *vp) 
{
  struct RingModulatorData *rmd = (struct RingModulatorData*)vp;
  float x = static_cast<float>(v),
        y = (1.0f-rmd->mix)*x + rmd->mix*static_cast<float>(sin(rmd->argument))*x;
  rmd->argument += rmd->argument_inc;
  return static_cast<short>(y);
}