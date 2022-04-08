#ifndef	HELPER_H
#define HELPER_H

namespace ManCong
{
	namespace Memory
	{
		class Helper
		{
		public:
			template <class T>
			static T& GetElement(T* ptr, size_t x, size_t y, size_t width)
			{
				return const_cast<T&>(GetElement(static_cast<T const*>(ptr), x, y, width));
			}

			template <class T>
			static T const& GetElement(T const* ptr, size_t x, size_t y, size_t width)
			{
				return *(ptr + y * width + x);
			}

		private:
			Helper(void) {};
			~Helper(void) {};
		};
	}
}

#endif