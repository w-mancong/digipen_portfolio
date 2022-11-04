/*!*****************************************************************************
\file polynomial.tpp
\author Wong Man Cong
\par DP email: w.mancong\@digipen.edu
\par Course: HLP3
\par Section: A
\par Quiz 5
\date 4-11-2022
\brief
Templated class that provides functionalities to declaring and calculating
polynomials
*******************************************************************************/

namespace HLP3 
{
  /*!*****************************************************************************
      \brief To print out the polynomial
  *******************************************************************************/
  template< typename T, int N > 
  std::ostream& operator<<(std::ostream &out, Polynomial<T, N> const& pol) 
  {
    out << pol[0] << " ";
    for ( int i=1; i<=N; ++i ) 
    {
      if ( pol[i] != 0 ) 
      { // skip terms with zero coefficients
        if      ( pol[i] > 0 ) {  out << "+"; }
        if      ( pol[i] == 1 )  { }
        else if ( pol[i] == -1 ) { out << "-"; }
        else                     { out << pol[i] << "*"; }
        out << "x^" << i << " ";
      }
    }
    return out;
  }

  template <typename T1, int N>
  Polynomial<T1, N>::Polynomial(void)
  {
    for (size_type i{}; i <= N; ++i)
      *(values + i) = static_cast<T1>(0);
  }

  template <typename T1, int N>
  template <typename T2>
  Polynomial<T1, N>::Polynomial(Polynomial<T2, N> const &rhs)
  {
    for (size_type i{}; i <= N; ++i)
      *(values + i) = static_cast<T1>(rhs[i]);
  }

  template <typename T1, int N>
  template <typename T2>
  Polynomial<T1, N> &Polynomial<T1, N>::operator=(Polynomial<T2, N> const &rhs)
  {
    Polynomial<T1, N> res{ rhs };
    swap(res);
    return *this;
  }

  template <typename T1, int N>
  typename Polynomial<T1, N>::value_type Polynomial<T1, N>::operator()(int a)
  {
    value_type res = *values;
    for (size_type i{1}; i <= N; ++i)
      res += *(values + i) * static_cast<value_type>(pow(a, i));
    return res;
  }

  template <typename T1, int N>
  template <int M>
  Polynomial<T1, N + M> Polynomial<T1, N>::operator*(Polynomial<T1, M> const &rhs)
  {
    Polynomial<T1, N + M> res{};

    for (size_type i{}; i <= N; ++i)
    {
      for (size_type j{}; j <= M; ++j)
        res[i + j] += (*this)[i] * rhs[j];
    }
    return res;
  }

  template <typename T1, int N>
  typename Polynomial<T1, N>::reference Polynomial<T1, N>::operator[](size_type index)
  {
    return const_cast<reference>(const_cast<Polynomial<T1, N> const &>(*this)[index]);
  }

  template <typename T1, int N>
  typename Polynomial<T1, N>::const_reference Polynomial<T1, N>::operator[](size_type index) const
  {
    // error checking if index is more than N
    return *(values + index);
  }

  template <typename T1, int N>
  void Polynomial<T1, N>::swap(Polynomial<T1, N> &rhs)
  {
    std::swap(values, rhs.values);
  }

  template <typename T1, int N>
  typename Polynomial<T1, N>::value_type Polynomial<T1, N>::pow(value_type base, size_type exponent)
  {
    value_type res{ static_cast<value_type>(1) };
    for (; exponent > 0; --exponent)
      res *= base;
    return res;
  }
} // end namespace HLP3
