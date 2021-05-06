#ifndef _CSR_
#define _CSR_

#include <vector>
#include <tuple>
#include <algorithm>



/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_rows      - number of rows in A
 *   I  n_cols      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note: 
 *   Input:  row and column indices *are not* assumed to be ordered
 *           
 *   Note: duplicate entries are carried over to the CSR represention
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_rows,n_cols))
 * 
 */
template <class I, class T>
void coo_tocsr(const I n_rows,
               const I n_cols,
               const I nnz,
               const I Ai[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    //compute number of non-zero entries per row of A 
    std::fill(Bp, Bp + n_rows, 0);

    for (I n = 0; n < nnz; n++){            
        Bp[Ai[n]]++;
    }

    //cumsum the nnz per row to get Bp[]
    for(I i = 0, cumsum = 0; i < n_rows; i++){     
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_rows] = nnz; 

    //write Aj,Ax into Bj,Bx
    for(I n = 0; n < nnz; n++){
        I row  = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for(I i = 0, last = 0; i <= n_rows; i++){
        I temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

/*
 * Sum together duplicate column entries in each row of CSR matrix A
 *
 *
 * Input Arguments:
 *   I    n_rows       - number of rows in A (and B)
 *   I    n_cols       - number of columns in A (and B)
 *   I    Ap[n_rows+1] - row pointer
 *   I    Aj[nnz(A)]  - column indices
 *   T    Ax[nnz(A)]  - nonzeros
 *
 * Note:
 *   The column indices within each row must be in sorted order.
 *   Explicit zeros are retained.
 *   Ap, Aj, and Ax will be modified *inplace*
 *
 */
template <class I, class T>
void csr_sum_duplicates(const I n_rows,
                        const I n_cols,
                              I Ap[],
                              I Aj[],
                              T Ax[])
{
    I nnz = 0;
    I row_end = 0;
    for(I i = 0; i < n_rows; i++){
        I jj = row_end;
        row_end = Ap[i+1];
        while( jj < row_end ){
            I j = Aj[jj];
            T x = Ax[jj];
            jj++;
            while( jj < row_end && Aj[jj] == j ){
                x += Ax[jj];
                jj++;
            }
            Aj[nnz] = j;
            Ax[nnz] = x;
            nnz++;
        }
        Ap[i+1] = nnz;
    }
}

/*
 * Eliminate zero entries from CSR matrix A
 *
 *
 * Input Arguments:
 *   I    n_rows       - number of rows in A (and B)
 *   I    n_cols       - number of columns in A (and B)
 *   I    Ap[n_rows+1] - row pointer
 *   I    Aj[nnz(A)]  - column indices
 *   T    Ax[nnz(A)]  - nonzeros
 *
 * Note:
 *   Ap, Aj, and Ax will be modified *inplace*
 *
 */
template <class I, class T>
void csr_eliminate_zeros(const I n_rows,
                         const I n_cols,
                               I Ap[],
                               I Aj[],
                               T Ax[])
{
    I nnz = 0;
    I row_end = 0;
    for(I i = 0; i < n_rows; i++){
        I jj = row_end;
        row_end = Ap[i+1];
        while( jj < row_end ){
            I j = Aj[jj];
            T x = Ax[jj];
            if(x != 0){
                Aj[nnz] = j;
                Ax[nnz] = x;
                nnz++;
            }
            jj++;
        }
        Ap[i+1] = nnz;
    }
}



template< class T1, class T2 >
bool kv_pair_less(const std::pair<T1,T2>& x, const std::pair<T1,T2>& y){
    return x.first < y.first;
}

/*
 * Sort CSR column indices inplace
 *
 * Input Arguments:
 *   I  n_rows           - number of rows in A
 *   I  Ap[n_rows+1]     - row pointer
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros
 *
 */
template<class I, class T>
void csr_sort_indices(const I n_rows,
                      const I Ap[],
                            I Aj[],
                            T Ax[])
{
    std::vector< std::pair<I,T> > temp;

    for(I i = 0; i < n_rows; i++){
        I row_start = Ap[i];
        I row_end   = Ap[i+1];

        temp.resize(row_end - row_start);
        for (I jj = row_start, n = 0; jj < row_end; jj++, n++){
            temp[n].first  = Aj[jj];
            temp[n].second = Ax[jj];
        }

        std::sort(temp.begin(),temp.end(),kv_pair_less<I,T>);

        for(I jj = row_start, n = 0; jj < row_end; jj++, n++){
            Aj[jj] = temp[n].first;
            Ax[jj] = temp[n].second;
        }
    }
}


/*
 * Determine whether the matrix structure is canonical CSR.
 * Canonical CSR implies that column indices within each row
 * are (1) sorted and (2) unique.  Matrices that meet these
 * conditions facilitate faster matrix computations.
 *
 * Input Arguments:
 *   I  n_rows           - number of rows in A
 *   I  Ap[n_rows+1]     - row pointer
 *   I  Aj[nnz(A)]      - column indices
 *
 */
template <class I>
bool csr_has_canonical_format(const I n_rows,
                              const I Ap[],
                              const I Aj[])
{
    for(I i = 0; i < n_rows; i++){
        if (Ap[i] > Ap[i+1])
            return false;
        for(I jj = Ap[i] + 1; jj < Ap[i+1]; jj++){
            if( !(Aj[jj-1] < Aj[jj]) ){
                return false;
            }
        }
    }
    return true;
}







template<class T,class I>
class csr_matrix
{
	typedef std::vector<T> vec_T;
	typedef std::vector<I> vec_I;

	vec_T data;
	vec_I indptr,indices;
	const I nr,nc;


public:
	csr_matrix(std::tuple<vec_T&,vec_I&,vec_I&> args,const I n_rows,const I n_cols) : nr(n_rows), nc(n_cols) {
		// assume COO format
		
		const I nnz = std::get<0>(args).size();
		data.resize(nnz);
		indices.resize(nnz);
		indptr.resize(nr+1);

		coo_tocsr(n_rows,n_cols,nnz,
			&std::get<1>(args)[0],
			&std::get<2>(args)[0],
			&std::get<0>(args)[0],
			&indptr[0],
			&indices[0],
			&data[0]);

		this->canonical_format();
	}

    csr_matrix(csr_matrix<T,I> &other): nr(other.n_rows()), nc(other.n_cols()){
        std::copy(other.indptr.begin(),other.indptr.end(),indptr.begin());
        std::copy(other.indices.begin(),other.indices.end(),indices.begin());
        std::copy(other.data.begin(),other.data.end(),data.begin());

    }
	~csr_matrix() {}

	void canonical_format(){
		if(!csr_has_canonical_format(nr,&indptr[0],&indices[0])){
			csr_sum_duplicates(nr,nc,&indptr[0],&indices[0],&data[0]);
			csr_sum_duplicates(nr,nc,&indptr[0],&indices[0],&data[0]);
			csr_sort_indices(nr,&indptr[0],&indices[0],&data[0]);

			const I nnz = indptr[nr];

			indices.resize(nnz);
			data.resize(nnz);
		}
	}


	template<class in,class out>
	void dot(in &Xx,out &Yx){
		for(I i = 0; i < nr; i++){
			auto sum = ((T)0) * Xx[i];
			for(I jj = indptr[i]; jj < indptr[i+1]; jj++){
				sum += data[jj] * Xx[indices[jj]];
			}
			Yx[i] = sum;
		}
	}
	
    I n_rows() const {return nr;}
    I n_cols() const {return nc;}
    I nnz() const {return indptr[nr];}

};








#endif