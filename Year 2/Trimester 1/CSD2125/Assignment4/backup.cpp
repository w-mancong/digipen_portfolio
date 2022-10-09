// provide Doxygen file header documentation blocks
// provide Doxygen function header documentation block
#include "wave.h"
#include <fstream>
#include <climits>
#include <cstring>

struct WaveData_tag 
{
  int channel_count{};
  int frame_count{};
  int sampling_rate{};
  void *filter_data{};
  short *data16{};
};

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

// API functions
char *waveRead(char const *filename)
{
  // Initialising input file stream
  std::ifstream ifs(filename, std::ios::binary);
  ifs.seekg(0, ifs.end);
  // Getting the size of the file in bytes
  size_t const len{ static_cast<size_t>(ifs.tellg()) };
  ifs.seekg(0, ifs.beg);

  if(!ifs)
    return nullptr;

  // Create a memory pool of char*
  char *fileContents = new char[len];
  ifs.read(fileContents, len);
  return fileContents;
}

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

void waveWrite(WaveData w, char const *filename)
{
  std::ofstream ofs(filename, std::ios::binary);
  ofs.write(reinterpret_cast<char *>(&wave_header), sizeof(WaveHeader));
  ofs.write(reinterpret_cast<char *>(w->data16), sizeof(short) * static_cast<size_t>(w->channel_count) * static_cast<size_t>(w->frame_count));
}

void waveDestroy(WaveData w)
{
  delete[] w->data16;
  delete w;
}

int waveChannelCount(WaveDataReadOnly w)
{
  return w->channel_count;
}

int waveFrameCount(WaveDataReadOnly w)
{
  return w->frame_count;
}

int waveSamplingRate(WaveDataReadOnly w)
{
  return w->sampling_rate;
}

void waveSetFilterData(WaveData w, void *vp)
{
  w->filter_data = vp;
}

void waveFilter(WaveData w, int channel, short (*filter)(short, void *))
{
  for (int i{}; i < w->frame_count; ++i)
    *(w->data16 + (w->channel_count * i) + channel) = filter(waveRetrieveSample(w, i, channel), w->filter_data);
}

short waveRetrieveSample(WaveDataReadOnly w, int frame, int channel)
{
  return *(w->data16 + (w->channel_count * frame) + channel);
}

// filter functions used in calls to waveFilte
short waveCut6DB(short in, void *data)
{
  (void)data;
  return in >> 1;
}

short waveBoost3DB(short in, void *data)
{
  (void)data;
  int ix = static_cast<int>(in);
  int output = ix + (ix >> 2) + (ix >> 3) + (ix >> 5);

  if (output > std::numeric_limits<short>::max())
    output = std::numeric_limits<short>::max();
  else if (output > std::numeric_limits<short>::min())
    output = std::numeric_limits<short>::min();

  return static_cast<short>(output);
}

// the filter function provided to you ...
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

short waveBoost(short in, void *data)
{
  int res{};
  short d{ *reinterpret_cast<short*>(data) };

  /*
    return bit based on the current position
    7 6 5 4 3 2 1 0  <- Bit position
    1 0 0 1 0 0 0 1  <- Binary data

    if pos is 3, return value (rv) is 0
    if pos is 7, rv is 1
  */
  auto get_bit = [&](size_t pos)
  {
    return d & (0b1 << pos);
  };

  for (size_t i{}; i < 16; ++i)
  {
    if(!get_bit(i))
      continue;
    res += static_cast<int>(in << i);
  }

  // Overflowed, cast it to it's short max and min
  if(res > std::numeric_limits<short>::max())
    res = std::numeric_limits<short>::max();
  else if(res < std::numeric_limits<short>::min())
    res = std::numeric_limits<short>::min();

  return static_cast<short>(res);
}
