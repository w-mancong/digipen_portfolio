// Provide file documentation header
// Don't include any library headers!!!

namespace HLP3 
{
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

  template <typename T1, size_t N>
  Polynomial<T1, N>::Polynomial(void) : values{ new T1[N] }
  {
    for (size_type i{}; i < N; ++i)
      *(values + i) = static_cast<T1>(1);
  }

  template <typename T1, size_t N>
  Polynomial<T1, N>::~Polynomial(void)
  {
    delete[] values;
  }

  template <typename T1, size_t N>
  template <typename T2>
  Polynomial<T1, N>::Polynomial(Polynomial<T2, N> const &rhs)
  {

  }

  template <typename T1, size_t N>
  template <typename T2>
  Polynomial<T1, N>& Polynomial<T1, N>::operator=(Polynomial<T2, N> const &rhs)
  {

  }

  template <typename T1, size_t N>
  typename Polynomial<T1, N>::value_type Polynomial<T1, N>::operator()(int a)
  {

  }

  template <typename T1, size_t N>
  template <size_t M>
  Polynomial<T1, N + M> Polynomial<T1, N>::operator*(Polynomial<T1, M> const &rhs)
  {

  }

  template <typename T1, size_t N>
  typename Polynomial<T1, N>::reference Polynomial<T1, N>::operator[](size_type index)
  {

  }

  template <typename T1, size_t N>
  typename Polynomial<T1, N>::const_reference Polynomial<T1, N>::operator[](size_type index) const
  {

  }
} // end namespace HLP3
