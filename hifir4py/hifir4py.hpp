///////////////////////////////////////////////////////////////////////////////
//              This file is part of HIFIR4PY project                        //
//                                                                           //
//     Copyright (C) 2019--2021 NumGeom Group at Stony Brook University      //
//                                                                           //
//     This program is free software: you can redistribute it and/or modify  //
//     it under the terms of the GNU Affero General Public License as published //
//     by the Free Software Foundation, either version 3 of the License, or  //
//     (at your option) any later version.                                   //
//                                                                           //
//     This program is distributed in the hope that it will be useful,       //
//     but WITHOUT ANY WARRANTY; without even the implied warranty of        //
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
//     GNU Affero General Public License for more details.                   //
//                                                                           //
//     You should have received a copy of the GNU Affero General Public License //
//     along with this program.  If not, see <https://www.gnu.org/licenses/>.//
///////////////////////////////////////////////////////////////////////////////

// Authors:
//  Qiao,

#ifndef _HIFIR_PYTHON_HIFIR4PY_HPP
#define _HIFIR_PYTHON_HIFIR4PY_HPP

// NOTE we need to ensure C++ code throws exceptions
#ifndef HIF_THROW
#  define HIF_THROW
#endif  // HIF_THROW

// we need to make sure that the stdout and stderr are not pre-defined
#ifdef HIF_STDOUT
#  undef HIF_STDOUT
#endif  // HIF_STDOUT
#ifdef HIF_STDERR
#  undef HIF_STDERR
#endif  // HIF_STDOUT

#include <string>
#include <type_traits>
#include <vector>

// then include the file descriptor wrapper
#include "file_dp_api.h"

// stdout wrapper
#define HIF_STDOUT(__msg)       \
  do {                          \
    import_hifir4py__file_dp(); \
    hifir4py_stdout(__msg);     \
  } while (false)

// stderr wrapper
#define HIF_STDERR(__msg)       \
  do {                          \
    import_hifir4py__file_dp(); \
    hifir4py_stderr(__msg);     \
  } while (false)

// now, include the hifir code generator
#include <HIF.hpp>
#include "hifir4py_nsp.hpp"
namespace hif {

// types
using py_crs_type   = DefaultHIF::crs_type;
using py_array_type = DefaultHIF::array_type;
using size_type     = DefaultHIF::size_type;

// read native hifir format
inline void read_hifir(const std::string &fn, std::size_t &nrows,
                       std::size_t &ncols, std::size_t &m,
                       std::vector<int> &indptr, std::vector<int> &indices,
                       std::vector<double> &vals, const bool is_bin = true) {
  Array<int>    ind_ptr, inds;
  Array<double> values;
  auto          A = is_bin ? py_crs_type::from_bin(fn.c_str(), &m)
                  : py_crs_type::from_ascii(fn.c_str(), &m);
  nrows = A.nrows();
  ncols = A.ncols();
  ind_ptr.swap(A.row_start());
  inds.swap(A.col_ind());
  values.swap(A.vals());
  // efficient for c++11
  indptr  = std::vector<int>(ind_ptr.cbegin(), ind_ptr.cend());
  indices = std::vector<int>(inds.cbegin(), inds.cend());
  vals    = std::vector<double>(values.cbegin(), values.cend());
}

// write native hifir format
inline void write_hifir(const std::string &fn, const std::size_t nrows,
                        const std::size_t ncols, const int *indptr,
                        const int *indices, const double *vals,
                        const std::size_t m0, const bool is_bin = true) {
  constexpr static bool WRAP = true;
  const py_crs_type     A(nrows, ncols, const_cast<int *>(indptr),
                          const_cast<int *>(indices), const_cast<double *>(vals),
                          WRAP);
  // aggressively checking
  A.check_validity();
  if (is_bin)
    A.write_bin(fn.c_str(), m0);
  else
    A.write_ascii(fn.c_str(), m0);
}

// query file information
inline void query_hifir_info(const std::string &fn, bool &is_row, bool &is_c,
                             bool &is_double, bool &is_real,
                             std::uint64_t &nrows, std::uint64_t &ncols,
                             std::uint64_t &nnz, std::uint64_t &m,
                             const bool is_bin = true) {
  std::tie(is_row, is_c, is_double, is_real, nrows, ncols, nnz, m) =
      is_bin ? query_info_bin(fn.c_str()) : query_info_ascii(fn.c_str());
}

// In order to make things easier, we directly use the raw data types, thus
// we need to create a child class.
class PyHIF : public DefaultHIF {
 public:
  using base = DefaultHIF;

  // factorize crs
  inline void factorize_raw(const size_type n, const int *rowptr,
                            const int *colind, const double *vals,
                            const size_type m0, const Options &opts) {
    constexpr static bool WRAP = true;

    py_crs_type A(n, n, const_cast<int *>(rowptr), const_cast<int *>(colind),
                  const_cast<double *>(vals), WRAP);
    base::factorize(A, opts, m0);
  }

  // solve
  inline void solve_raw(const size_type n, const double *b, double *x,
                        const bool      trans = false,
                        const size_type r     = 0u) const {
    constexpr static bool WRAP = true;

    const array_type B(n, const_cast<double *>(b), WRAP);
    array_type       X(n, x, WRAP);
    base::solve(B, X, trans, r);
  }

  // solve with iterative refinement
  inline void hifir_raw(const size_type n, const int *rowptr, const int *colind,
                        const double *vals, const double *b, const size_type N,
                        double *x, const bool trans = false,
                        const size_type r = static_cast<size_type>(-1)) const {
    constexpr static bool WRAP = true;

    const array_type  B(n, const_cast<double *>(b), WRAP);
    array_type        X(n, x, WRAP);
    const py_crs_type A(n, n, const_cast<int *>(rowptr),
                        const_cast<int *>(colind), const_cast<double *>(vals),
                        WRAP);
    base::hifir(A, B, N, X, trans, r);
  }

  // solve
  inline void mmultiply_raw(const size_type n, const double *b, double *x,
                            const bool      trans = false,
                            const size_type r     = 0u) const {
    constexpr static bool WRAP = true;

    const array_type B(n, const_cast<double *>(b), WRAP);
    array_type       X(n, x, WRAP);
    base::mmultiply(B, X, trans, r);
  }
};

// mixed precision, using float preconditioner
class PyHIF_Mixed : public HIF<float, int> {
 public:
  using base = HIF<float, int>;

  // factorize crs
  inline void factorize_raw(const size_type n, const int *rowptr,
                            const int *colind, const double *vals,
                            const size_type m0, const Options &opts) {
    constexpr static bool WRAP = true;

    py_crs_type A(n, n, const_cast<int *>(rowptr), const_cast<int *>(colind),
                  const_cast<double *>(vals), WRAP);
    base::factorize(A, opts, m0);
  }

  // solve
  inline void solve_raw(const size_type n, const double *b, double *x,
                        const bool      trans = false,
                        const size_type r     = 0u) const {
    constexpr static bool WRAP = true;

    const py_array_type B(n, const_cast<double *>(b), WRAP);
    py_array_type       X(n, x, WRAP);
    base::solve(B, X, trans, r);
  }

  // solve with iterative refinement
  inline void hifir_raw(const size_type n, const int *rowptr, const int *colind,
                        const double *vals, const double *b, const size_type N,
                        double *x, const bool trans = false,
                        const size_type r = static_cast<size_type>(-1)) const {
    constexpr static bool WRAP = true;

    const py_array_type B(n, const_cast<double *>(b), WRAP);
    py_array_type       X(n, x, WRAP);
    const py_crs_type   A(n, n, const_cast<int *>(rowptr),
                          const_cast<int *>(colind), const_cast<double *>(vals),
                          WRAP);
    base::hifir(A, B, N, X, trans, r);
  }

  // solve
  inline void mmultiply_raw(const size_type n, const double *b, double *x,
                            const bool      trans = false,
                            const size_type r     = 0u) const {
    constexpr static bool WRAP = true;

    const py_array_type B(n, const_cast<double *>(b), WRAP);
    py_array_type       X(n, x, WRAP);
    base::mmultiply(B, X, trans, r);
  }
};

// abstract interface for solver
class KspSolver {
 public:
  virtual ~KspSolver() {}
  virtual void                      set_rtol(const double)           = 0;
  virtual double                    get_rtol() const                 = 0;
  virtual void                      set_restart(const int)           = 0;
  virtual int                       get_restart() const              = 0;
  virtual void                      set_maxit(const size_type)       = 0;
  virtual size_type                 get_maxit() const                = 0;
  virtual void                      set_inner_steps(const size_type) = 0;
  virtual size_type                 get_inner_steps() const          = 0;
  virtual void                      set_lamb1(const double)          = 0;
  virtual double                    get_lamb1() const                = 0;
  virtual void                      set_lamb2(const double)          = 0;
  virtual double                    get_lamb2() const                = 0;
  virtual void                      check_pars()                     = 0;
  virtual int                       get_resids_length() const        = 0;
  virtual void                      get_resids(double *r) const      = 0;
  virtual std::pair<int, size_type> solve_raw(
      const size_type n, const int *rowptr, const int *colind,
      const double *vals, const double *b, double *x, const int kernel,
      const bool with_init_guess, const bool verbose) const = 0;
};

// using a template base for Ksp solver
template <template <class, class> class Ksp, class MType = PyHIF,
          class ValueType = void>
class KspAdapt : public Ksp<MType, ValueType>, public KspSolver {
 public:
  using base = Ksp<MType, ValueType>;
  virtual ~KspAdapt() {}

  virtual void   set_rtol(const double v) override final { base::rtol = v; }
  virtual double get_rtol() const override final { return base::rtol; }
  virtual void   set_restart(const int v) override final { base::restart = v; }
  virtual int    get_restart() const override final { return base::restart; }
  virtual void set_maxit(const size_type v) override final { base::maxit = v; }
  virtual size_type get_maxit() const override final { return base::maxit; }
  virtual void      set_inner_steps(const size_type v) override final {
    base::inner_steps = v;
  }
  virtual size_type get_inner_steps() const override final {
    return base::inner_steps;
  }
  virtual void   set_lamb1(const double v) override final { base::lamb1 = v; }
  virtual double get_lamb1() const override final { return base::lamb1; }
  virtual void   set_lamb2(const double v) override final { base::lamb2 = v; }
  virtual double get_lamb2() const override final { return base::lamb2; }

  virtual int get_resids_length() const override final {
    return base::_resids.size();
  }
  virtual void get_resids(double *r) const override final {
    for (int i = 0; i < get_resids_length(); ++i) r[i] = base::_resids[i];
  }

  virtual void check_pars() override final { base::_check_pars(); }

  virtual std::pair<int, size_type> solve_raw(
      const size_type n, const int *rowptr, const int *colind,
      const double *vals, const double *b, double *x, const int kernel,
      const bool with_init_guess, const bool verbose) const override final {
    constexpr static bool WRAP = true;

    const py_crs_type A(n, n, const_cast<int *>(rowptr),
                        const_cast<int *>(colind), const_cast<double *>(vals),
                        WRAP);
#ifndef NDEBUG
    A.check_validity();
#endif
    const py_array_type bb(n, const_cast<double *>(b), WRAP);
    py_array_type       xx(n, x, WRAP);
    return base::solve(A, bb, xx, kernel, with_init_guess, verbose);
  }
};

using PyGMRES            = KspAdapt<ksp::GMRES>;       // gmres
using PyFGMRES           = KspAdapt<ksp::FGMRES>;      // fgmres
using PyFQMRCGSTAB       = KspAdapt<ksp::FQMRCGSTAB>;  // fqmrcgstab
using PyFBICGSTAB        = KspAdapt<ksp::FBICGSTAB>;   // fbicgstab
using PyTGMRESR          = KspAdapt<ksp::TGMRESR>;
using PyGMRES_Mixed      = KspAdapt<ksp::GMRES, PyHIF_Mixed, double>;
using PyFGMRES_Mixed     = KspAdapt<ksp::FGMRES, PyHIF_Mixed, double>;
using PyFQMRCGSTAB_Mixed = KspAdapt<ksp::FQMRCGSTAB, PyHIF_Mixed, double>;
using PyFBICGSTAB_Mixed  = KspAdapt<ksp::FBICGSTAB, PyHIF_Mixed, double>;
using PyTGMRESR_Mixed    = KspAdapt<ksp::TGMRESR, PyHIF_Mixed, double>;

}  // namespace hif

#endif  // _HIFIR_PYTHON_HIFIR4PY_HPP
