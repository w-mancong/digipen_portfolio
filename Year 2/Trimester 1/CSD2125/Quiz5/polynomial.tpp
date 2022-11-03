// Provide file documentation header
// Don't include any library headers!!!

namespace HLP3 
{

// Define member functions of class template Polynomial ...

// DON'T CHANGE/EDIT THE FOLLOWING DEFINITION:
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



} // end namespace HLP3
