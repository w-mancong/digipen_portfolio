#include "string.h"
#include <algorithm>
#include <cassert>

namespace
{
	my_std::string strcat(my_std::string const& lhs, my_std::string const& rhs)
	{
		my_std::string ret{ lhs };
		ret += rhs;
		return ret;
	}
}

namespace my_std
{
	string::string(void) : len{ 0 }, str{ new value_type{'\0'} }
	{

	}

	string::string(const_pointer str) : len{ size(str) }, str{ new value_type[len + 1] }
	{
		strcpy(this->str, str);
	}

	string::string(value_type ch) : len{ 1 }, str{ new value_type[len + 1] }
	{
		*str = ch;
		*(str + 1) = '\0';
	}

	string::string(std::initializer_list<value_type> list) : len{ list.size() }, str{ new value_type[len + 1] }
	{
		pointer ptr = str;
		for (value_type ch : list)
			*ptr++ = ch;
		*ptr = '\0';
	}

	string::~string(void)
	{
		clean(true);
	}

	// COPY
	string::string(string const& rhs) noexcept : len{ size(rhs.str) }, str{ new value_type[len + 1] }
	{
		strcpy(this->str, rhs.str);
	}

	string& string::operator=(string const& rhs) noexcept
	{
		string temp{ rhs };
		swap(temp);
		return *this;
	}

	// MOVE
	string::string(string&& rhs) noexcept
	{
		move(rhs);
	}

	string& string::operator=(string&& rhs) noexcept
	{
		move(rhs);
		return *this;
	}

	string& string::operator+=(string const& rhs)
	{
		strcat(rhs.str);
		return *this;
	}

	string& string::operator+=(const_pointer rhs)
	{
		strcat(rhs);
		return *this;
	}

	string& string::operator+=(value_type rhs)
	{
		strcat(rhs);
		return *this;
	}

	string::const_reference string::operator[](size_type index) const
	{
		return get(index);
	}

	string::reference string::operator[](size_type index)
	{
		return const_cast<reference>(get(index));
	}

	string::const_reference string::front(void) const
	{
		return get(0);
	}

	string::reference string::front(void)
	{
		return const_cast<reference>(get(0));
	}

	string::const_reference string::back(void) const
	{
		return get(len - 1);
	}

	string::reference string::back(void)
	{
		return const_cast<reference>(get(len - 1));
	}

	string::const_reference string::at(size_type index) const
	{
		return get(index);
	}

	string::reference string::at(size_type index)
	{
		return const_cast<reference>(get(index));
	}

	string::const_pointer string::c_str(void) const
	{
		return const_cast<const_pointer>(str);
	}

	string::size_type string::size(void) const
	{
		return len;
	}

	// return true if it's empty
	bool string::empty(void) const
	{
		return !len;
	}

	void string::clear(void)
	{
		clean();
	}

	string::size_type string::size(const_pointer str) const
	{
		const_pointer ptr = str;
		while (*ptr) ++ptr;
		return ptr - str;
	}

	void string::strcpy(pointer dst, const_pointer src)
	{
		while (*src)
			*(dst)++ = *(src)++;
		*dst = '\0';
	}

	// resize and concatentonate
	void string::strcat(string const& src)
	{
		// create a temporary copy of str
		string temp{ str };
		// new size
		len = temp.size() + src.size();	
		// delete old memory location
		delete[] str;
		// allocate new memory location for str
		str = new value_type[len + 1];
		pointer ptr = str;
		const_pointer old_ptr = temp.str, src_ptr = src.str;
		// copy contents of old string into str
		while (*old_ptr)
			*(ptr)++ = *(old_ptr)++;
		// copy contents of src into str
		while (*src_ptr)
			*(ptr)++ = *(src_ptr)++;
		*ptr = '\0';
	}

	void string::swap(string& temp)
	{
		std::swap(len, temp.len);
		std::swap(str, temp.str);
	}

	const char& string::get(size_type index) const
	{
		assert(index < len);
		return *(str + index);
	}

	void string::clean(bool des)
	{
		if (str)
		{
			delete[] str;
			str = nullptr;
		}
		if (!des)
			str = new char{ '\0' };
		len = 0;
	}

	void string::move(string& rhs)
	{
		len = rhs.len;
		rhs.len = 0;
		str = rhs.str;
		rhs.str = new char{ '\0' };
	}

	string operator+(string const& lhs, string const& rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(string const& lhs, const char* rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(const char* lhs, string const& rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(string const& lhs, char rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(char lhs, string const& rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(string& lhs, char rhs)
	{
		return strcat(lhs, rhs);
	}

	string operator+(char lhs, string& rhs)
	{
		return strcat(lhs, rhs);
	}

	bool operator==(string const& lhs, string const& rhs)
	{
		if (lhs.size() < rhs.size())
			return false;
		for (size_t i = 0; i < lhs.size(); ++i)
		{
			if (lhs[i] != rhs[i])
				return false;
		}
		return true;
	}

	bool operator<(string const& lhs, string const& rhs)
	{
		const size_t SHORTEST_LEN = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
		for (size_t i = 0; i < SHORTEST_LEN; ++i)
		{
			if (lhs[i] < rhs[i])
				return true;
			if (lhs[i] != rhs[i])
				return false;
		}
		if (lhs.size() < rhs.size())
			return true;
		return false;
	}

	bool operator>(string const& lhs, string const& rhs)
	{
		const size_t SHORTEST_LEN = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
		for (size_t i = 0; i < SHORTEST_LEN; ++i)
		{
			if (lhs[i] > rhs[i])
				return true;
			if (lhs[i] != rhs[i])
				break;
		}
		return false;
	}

	std::ostream& operator<<(std::ostream& os, string const& rhs)
	{
		for (size_t i = 0; i < rhs.size(); ++i)
			os << rhs[i];
		return os;
	}

	std::istream& operator>>(std::istream& is, string& rhs)
	{
		while (is.peek() != ' ' && is.peek() != '\n') rhs += static_cast<char>(is.get());
		return is;
	}
}