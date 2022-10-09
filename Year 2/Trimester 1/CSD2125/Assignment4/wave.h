/* wave.h
** -- simple interface for WAVE file functions
** cs225 5/25/2021
*/

#ifndef CS225_WAVE_H
#define CS225_WAVE_H

//typedef struct WaveData_tag* WaveData; // this is what you'd do in C
using WaveData = struct WaveData_tag*; // this is the modern C++ way
//typedef struct WaveData_tag const* WaveDataReadOnly; // this is C's way
using WaveDataReadOnly = struct WaveData_tag const*; // this is the modern C++ way

// API functions
char *waveRead(char const *filename);
WaveData waveParse(char const *file_contents);
void waveWrite(WaveData w, char const *filename);
void waveDestroy(WaveData w);
int waveChannelCount(WaveDataReadOnly w);
int waveFrameCount(WaveDataReadOnly w);
int waveSamplingRate(WaveDataReadOnly w);
void waveSetFilterData(WaveData w, void *vp);
void waveFilter(WaveData w, int channel, short (*filter)(short, void*));
short waveRetrieveSample(WaveDataReadOnly w, int frame, int channel);

// filter functions used in calls to waveFilter
short waveCut6DB(short in, void *data);
short waveBoost3DB(short in, void *data);
unsigned short waveBoostData(float gain);
short waveBoost(short in, void *data);

#endif