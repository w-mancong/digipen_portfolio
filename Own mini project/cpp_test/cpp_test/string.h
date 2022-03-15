#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <initializer_list>

namespace my_std
{
	class string
	{
	private:
		using value_type		= char;
		using pointer			= value_type*;
		using const_pointer		= const value_type*;
		using reference			= value_type&;
		using const_reference	= const value_type&;
		using size_type			= size_t;

	public:
		string(void);
		string(const_pointer str);
		string(value_type ch);
		string(std::initializer_list<value_type> list);
		~string(void);

		// COPY
		string(string const& rhs) noexcept;
		string& operator=(string const& rhs) noexcept;

		// MOVE
		string(string&& rhs) noexcept;
		string& operator=(string&& rhs) noexcept;

		string& operator+=(string const& rhs);
		string& operator+=(const_pointer rhs);
		string& operator+=(value_type rhs);

		const_reference operator[](size_type index) const;
		reference		operator[](size_type index);

		const_reference front(void) const;
		reference		front(void);

		const_reference back(void) const;
		reference		back(void);

		const_reference at(size_type index) const;
		reference		at(size_type index);

		const_pointer	c_str(void) const;

		size_type		size(void) const;

		bool			empty(void) const;

		void			clear(void);

	private:
		size_type size(const_pointer str) const;
		void strcpy(pointer dst, const_pointer src);
		void strcat(string const& src);
		void swap(string& temp);
		const char& get(size_type index) const;
		void clean(bool des = false);
		void move(string& rhs);

		size_type len;
		pointer str;
	};

	string operator+(string const& lhs, string const& rhs);

	string operator+(string const& lhs, const char* rhs);
	string operator+(const char* lhs, string const& rhs);

	string operator+(string const& lhs, char rhs);
	string operator+(char lhs, string const& rhs);

	bool operator==(string const& lhs, string const& rhs);
	bool operator<(string const& lhs, string const& rhs);
	bool operator>(string const& lhs, string const& rhs);

	std::ostream& operator<<(std::ostream& os, string const& rhs);
	std::istream& operator>>(std::istream& is, string& rhs);
}

#endif