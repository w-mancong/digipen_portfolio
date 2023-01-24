/*!*****************************************************************************
\file wave.cpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Assignment 4
\date 02-10-2022
\brief
This file contains function definition to read in a .wav audio file and
manipulate it's data, then storing it back into a .wav audio file
*******************************************************************************/
#include "wave.h"
#include <fstream>
#include <climits>
#include <cstring>

/*!*****************************************************************************
    \brief
    Structure to store some information about the WaveHeader, this struct
    is mainly used for this assignment's purposes
*******************************************************************************/
struct WaveData_tag
{
  int channel_count{};
  int frame_count{};
  int sampling_rate{};
  void *filter_data{};
  short *data16{};
};

/*!*****************************************************************************
    \brief
    This struct contains information for WaveHeader
*******************************************************************************/
struct WaveHeader
{
  char riff_label[4];              // (offset 0) = {'R','I','F','F'}
  unsigned riff_size;              // (offset 4) = 36 + data_size
  char file_tag[4];                // (offset 8) = {'W','A','V','E'}
  char fmt_label[4];               // (offset 12) = {'f','m','t',' '}
  unsigned fmt_size;               // (offset 16) = 16
  unsigned short audio_format;     // (offset 20) = 1
  unsigned short channel_count;    // (offset 22) = 1 or 2
  unsigned sampling_rate;          // (offset 24) = <anything>
  unsigned bytes_per_second;       // (offset 28) = <see below>
  unsigned short bytes_per_sample; // (offset 32) = <see below>
  unsigned short bits_per_sample;  // (offset 34) = 16
  char data_label[4];              // (offset 36) = {'d','a','t','a'}
  unsigned data_size;              // (offset 40) = <# of bytes of data>
} wave_header;

using wdt = WaveData_tag;

/*!*****************************************************************************
    \brief Reads wave file in as binary and store them into a dynamic array of chars

    \param [in] filename: Name of the wave file to be read

    \return Pointer to the first element of char storing the data of the wave file read
*******************************************************************************/
char *waveRead(char const *filename)
{
  // Initialising input file stream
  std::ifstream ifs(filename, std::ios::binary);
  ifs.seekg(0, ifs.end);
  // Getting the size of the file in bytes
  size_t const len{static_cast<size_t>(ifs.tellg())};
  ifs.seekg(0, ifs.beg);

  if (!ifs)
    return nullptr;

  // Create a memory pool of char*
  char *fileContents = new char[len];
  ifs.read(fileContents, len);
  return fileContents;
}

/*!*****************************************************************************
    \brief Parsing the data store inside file_contents. file_contents
    will be deallocated in this function

    \param [in] file_contents: Pointer to the first element of char allocated
    when using waveRead

    \return Pointer to the first element of WaveData
*******************************************************************************/
WaveData waveParse(char const *file_contents)
{
  if (!file_contents)
    return nullptr;
  // retrieving the data of the file header
  memcpy(&wave_header, file_contents, sizeof(WaveHeader));
  // Check if bits is 16 pits
  if (wave_header.bits_per_sample != 16)
    return nullptr;

  // Allocating memory for WaveData
  WaveData wd = new wdt;
  wd->channel_count = wave_header.channel_count;
  wd->sampling_rate = wave_header.sampling_rate;
  wd->frame_count = wave_header.data_size >> 1;
  // If it's stereo channel, frame count will be another half
  if (wd->channel_count == 2)
    wd->frame_count >>= 1;
  size_t const size = static_cast<size_t>(wave_header.data_size >> 1);
  wd->data16 = new short[size]{};
  memcpy(wd->data16, (file_contents + 44), sizeof(short) * size);

  delete[] file_contents;
  return wd;
}

/*!*****************************************************************************
    \brief Writing the contents of WaveData into an output file

    \param [in] w: Wave Data
    \param [in] filename: Name of the file to be stored into
*******************************************************************************/
void waveWrite(WaveData w, char const *filename)
{
  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<char *>(&wave_header), sizeof(WaveHeader));
  ofs.write(reinterpret_cast<char *>(w->data16), sizeof(short) * static_cast<size_t>(w->channel_count) * static_cast<size_t>(w->frame_count));
}

/*!*****************************************************************************
    \brief Deallocated memory for WaveData

    \param [in] w: The data to be deallocated
*******************************************************************************/
void waveDestroy(WaveData w)
{
  delete[] w->data16;
  delete w;
}

/*!*****************************************************************************
    \brief Returns the total number of channels

    \param [in] w: To retrieve channel count information from

    \return Total number of channels read by the .wav audio file
*******************************************************************************/
int waveChannelCount(WaveDataReadOnly w)
{
  return w->channel_count;
}

/*!*****************************************************************************
    \brief Return the total number of frame count

    \param [in] w: To retrieve frame count information from

    \return Total number of frame count
*******************************************************************************/
int waveFrameCount(WaveDataReadOnly w)
{
  return w->frame_count;
}

/*!*****************************************************************************
    \brief Return the sampling rate of of the audio file

    \param [in] w: To retrieve sampling rate information from

    \return The sampling rate of the .wav audio fle
*******************************************************************************/
int waveSamplingRate(WaveDataReadOnly w)
{
  return w->sampling_rate;
}

/*!*****************************************************************************
    \brief Set the filter data to be used

    \param [in] w: The data to be using the new filter data
    \param [in] vp: Address of the filter data
*******************************************************************************/
void waveSetFilterData(WaveData w, void *vp)
{
  w->filter_data = vp;
}

/*!*****************************************************************************
    \brief Filter audio data based on the channel and function

    \param [in] w: Audio data to be manipulated
    \param [in] channel: Channel of the sample to be manipulated
    \param [in] filter: Address of function pointer. This filter will manipulates 
    the data based on what the function is
*******************************************************************************/
void waveFilter(WaveData w, int channel, short (*filter)(short, void *))
{
  for (int i{}; i < w->frame_count; ++i)
    *(w->data16 + (w->channel_count * i) + channel) = filter(waveRetrieveSample(w, i, channel), w->filter_data);
}

/*!*****************************************************************************
    \brief Retrieve a particular audio sampled based on it's frame and channel

    \param [in] w: To retrieve the audio sample from
    \param [in] frame: Specifies which frame 
    \param [in] channel: Specifies which channel

    \return Audio sample based on the frame and channel
*******************************************************************************/
short waveRetrieveSample(WaveDataReadOnly w, int frame, int channel)
{
  return *(w->data16 + (w->channel_count * frame) + channel);
}

/*!*****************************************************************************
    \brief Multiply in by a scaling factor of 0.5

    \param [in] in: Audio sample to be manipulated
    \param [in] data: Parameter to be ignored

    \return The manipulated audio sample 
*******************************************************************************/
short waveCut6DB(short in, void *data)
{
  (void)data;
  return in >> 1;
}

/*!*****************************************************************************
    \brief Multiply in by a scaling factor of 1.41 (1.40625)

    \param [in] in: Audio sample to be manipulated
    \param [in] data: Parameter to be ignored

    \return The manipulated audio sample
*******************************************************************************/
short waveBoost3DB(short in, void *data)
{
  (void)data;
  int ix{ static_cast<int>(in) };
  int output = ix + (ix >> 2) + (ix >> 3) + (ix >> 5);

  if (output > std::numeric_limits<short>::max())
    output = std::numeric_limits<short>::max();
  else if (output < std::numeric_limits<short>::min())
    output = std::numeric_limits<short>::min();

  return static_cast<short>(output);
}

/*!*****************************************************************************
    \brief Convert gain into it's binary form

    \param [in] gain: Use to convert this floating point to it's binary form

    \return Gain in it's binary form
*******************************************************************************/
unsigned short waveBoostData(float gain)
{
  unsigned short bd = 0, mask = 0x01;
  for (int i = 0; i < 16; ++i)
  {
    if (gain >= 1)
    {
      bd |= mask;
      gain -= 1.0f;
    }
    mask <<= 1;
    gain *= 2.0f;
  }
  return bd;
}

/*!*****************************************************************************
    \brief Multiply in by a scaling factor of what data is

    \param [in] in: Audio sample to be manipulated
    \param [in] data: Pointer to the address of an unsigned short (will be using this as a scaling factor)

    \return The mainpulated audio sample
*******************************************************************************/
short waveBoost(short in, void *data)
{
  int output{}, ix{ static_cast<int>(in) };
  unsigned short val{*reinterpret_cast<unsigned short *>(data)};

  for (size_t i{}; i < 16; ++i, val >>= 1)
  {
    if(val & 0b1)
      output += ix >> i;
  }

  if (output > std::numeric_limits<short>::max())
    output = std::numeric_limits<short>::max();
  else if (output < std::numeric_limits<short>::min())
    output = std::numeric_limits<short>::min();

  return static_cast<short>(output);
}
