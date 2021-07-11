///////////////////////////////////////////////////////////////////////////////
//              This file is part of HIFIR4PY project                        //
///////////////////////////////////////////////////////////////////////////////

/*

    Copyright (C) 2019--2021 NumGeom Group at Stony Brook University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

// Authors:
//  Qiao,

#ifndef HIFIR4PY_HPP_
#define HIFIR4PY_HPP_

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

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

// The following is generated by Cython
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
#include "hifir.hpp"
namespace hifir4py {

constexpr static int NUM_PARAMS = _HIF_TOTAL_OPTIONS;
///< total number of parameters

const static std::vector<std::string> params_pos2tag = {
    "tau_L",     "tau_U",         "kappa_d", "kappa",     "alpha_L",
    "alpha_U",   "rho",           "c_d",     "c_h",       "N",
    "verbose",   "rf_par",        "reorder", "spd",       "check",
    "pre_scale", "symm_pre_lvls", "threads", "mumps_blr", "fat_schur_1st",
    "rrqr_cond", "pivot",         "gamma",   "beta",      "is_symm",
    "no_pre"};
///< position to tag mapping for \ref hif::Params

/*!
 * @brief Set the default parameters
 * @param[out] params Parameter array
 * @note \a params must at least has length of \ref NUM_PARAMS
 */
inline void set_default_params(double *params) {
  const auto &t2p                 = hif::internal::option_tag2pos;
  params[t2p.at("tau_L")]         = hif::DEFAULT_PARAMS.tau_L;
  params[t2p.at("tau_U")]         = hif::DEFAULT_PARAMS.tau_U;
  params[t2p.at("kappa_d")]       = hif::DEFAULT_PARAMS.kappa_d;
  params[t2p.at("kappa")]         = hif::DEFAULT_PARAMS.kappa;
  params[t2p.at("alpha_L")]       = hif::DEFAULT_PARAMS.alpha_L;
  params[t2p.at("alpha_U")]       = hif::DEFAULT_PARAMS.alpha_U;
  params[t2p.at("rho")]           = hif::DEFAULT_PARAMS.rho;
  params[t2p.at("c_d")]           = hif::DEFAULT_PARAMS.c_d;
  params[t2p.at("c_h")]           = hif::DEFAULT_PARAMS.c_h;
  params[t2p.at("N")]             = hif::DEFAULT_PARAMS.N;
  params[t2p.at("verbose")]       = hif::DEFAULT_PARAMS.verbose;
  params[t2p.at("rf_par")]        = hif::DEFAULT_PARAMS.rf_par;
  params[t2p.at("spd")]           = hif::DEFAULT_PARAMS.spd;
  params[t2p.at("check")]         = hif::DEFAULT_PARAMS.check;
  params[t2p.at("pre_scale")]     = hif::DEFAULT_PARAMS.pre_scale;
  params[t2p.at("symm_pre_lvls")] = hif::DEFAULT_PARAMS.symm_pre_lvls;
  params[t2p.at("threads")]       = hif::DEFAULT_PARAMS.threads;
  params[t2p.at("mumps_blr")]     = hif::DEFAULT_PARAMS.mumps_blr;
  params[t2p.at("fat_schur_1st")] = hif::DEFAULT_PARAMS.fat_schur_1st;
  params[t2p.at("rrqr_cond")]     = hif::DEFAULT_PARAMS.rrqr_cond;
  params[t2p.at("pivot")]         = hif::DEFAULT_PARAMS.pivot;
  params[t2p.at("gamma")]         = hif::DEFAULT_PARAMS.gamma;
  params[t2p.at("beta")]          = hif::DEFAULT_PARAMS.beta;
  params[t2p.at("is_symm")]       = hif::DEFAULT_PARAMS.is_symm;
  params[t2p.at("no_pre")]        = hif::DEFAULT_PARAMS.no_pre;
}

/*!
 * @class PyHIF
 * @tparam HifType A child type derived from \ref hif::HIF
 * @brief HIF class in Python
 */
template <class HifType>
class PyHIF : public HifType {
 protected:
  using _base = HifType;  ///< BaseType

 public:
  using value_type = typename _base::value_type;  ///< value type
  using index_type = typename _base::index_type;  ///< index type
  constexpr static bool IS_COMPLEX = !std::is_floating_point<value_type>::value;
  ///< whether or not using complex arithmetic
  constexpr static bool IS_MIXED = IS_COMPLEX
                                       ? sizeof(value_type) == sizeof(double)
                                       : sizeof(value_type) == sizeof(float);
  ///< whether or not using single precision for mixed-precision computation
  using interface_type =
      typename std::conditional<IS_COMPLEX, void, double>::type;
  ///< interface type in parameter, using void* for complex preconditioner
 protected:
  using _impl_type =
      typename std::conditional<IS_COMPLEX, std::complex<double>, double>::type;
  ///< implementation type used inside function bodies
  using _array_type  = hif::Array<_impl_type>;            ///< array
  using _matrix_type = hif::CRS<_impl_type, index_type>;  ///< matrix

 public:
  /*!
   * @brief Check mixed precision
   */
  static bool is_mixed() { return IS_MIXED; }

  /*!
   * @brief Check complex number
   */
  static bool is_complex() { return IS_COMPLEX; }

  /*!
   * @brief Check index size
   */
  static size_t index_size() { return sizeof(index_type); }

  /*!
   * @brief Compute the factorization
   * @param[in] n Input matrix size
   * @param[in] rowptr Row pointer in CRS
   * @param[in] colind Column indices in CRS
   * @param[in] vals Data values in CRS
   * @param[in] params Control parameters for HIF
   */
  inline void factorize(const size_t n, const index_type *rowptr,
                        const index_type *colind, const interface_type *vals,
                        const double *params = nullptr) {
    constexpr static bool WRAP = true;

    _matrix_type A(n, n, (index_type *)rowptr, (index_type *)colind,
                   (_impl_type *)vals, WRAP);
    // call factorization
    if (!params)
      _base::factorize(A);
    else {
      hif::Params pars;
      for (int i = 0; i < NUM_PARAMS; ++i)
        if (hif::set_option_attr<double>(params_pos2tag[i], params[i], pars))
          hif_error("failed to set %i parameter", i);
      _base::factorize(A, pars);
    }
  }

  /*!
   * @brief Perform multilevel triangular solve
   * @param[in] n Input vector size
   * @param[in] b Input RHS
   * @param[out] x Output solution
   * @param[in] trans Transpose/Hermitian tag (optional, default false)
   * @param[in] r Final Schur complement rank (optional)
   */
  inline void solve(const size_t n, const interface_type *b, interface_type *x,
                    const bool trans = false, const size_t r = 0u) const {
    constexpr static bool WRAP = true;

    const _array_type B(n, (_impl_type *)b, WRAP);
    _array_type       X(n, (_impl_type *)x, WRAP);
    _base::solve(B, X, trans, r);
  }

  /*!
   * @brief Perform multilevel matrix-vector multiplication
   * @param[in] n Input vector size
   * @param[in] b Input RHS
   * @param[out] x Output solution
   * @param[in] trans Transpose/Hermitian tag (optional, default false)
   * @param[in] r Final Schur complement rank (optional)
   */
  inline void mmultiply(const size_t n, const interface_type *b,
                        interface_type *x, const bool trans = false,
                        const size_t r = 0u) const {
    constexpr static bool WRAP = true;

    const _array_type B(n, (_impl_type *)b, WRAP);
    _array_type       X(n, (_impl_type *)x, WRAP);
    _base::mmultiply(B, X, trans, r);
  }

  /*!
   * @brief Perform multilevel triangular solve with iterative refinements
   * @param[in] n Input vector and matrix size
   * @param[in] rowptr Row pointer in CRS
   * @param[in] colind Column indices in CRS
   * @param[in] vals Data values in CRS
   * @param[in] b Input RHS
   * @param[in] nirs Number of IRs
   * @param[out] x Output solution
   * @param[in] trans Transpose/Hermitian tag (optional, default false)
   * @param[in] r Final Schur complement rank (optional)
   */
  inline void hifir(const size_t n, const index_type *rowptr,
                    const index_type *colind, const interface_type *vals,
                    const interface_type *b, const size_t nirs,
                    interface_type *x, const bool trans = false,
                    const size_t r = static_cast<size_t>(-1)) const {
    constexpr static bool WRAP = true;

    const _array_type B(n, (_impl_type *)b, WRAP);
    _array_type       X(n, (_impl_type *)x, WRAP);
    _matrix_type      A(n, n, (index_type *)rowptr, (index_type *)colind,
                   (_impl_type *)vals, WRAP);
    _base::hifir(A, B, nirs, X, trans, r);
  }

  /*!
   * @brief Perform multilevel triangular solve with IR and residual bounds
   * @param[in] n Input vector and matrix size
   * @param[in] rowptr Row pointer in CRS
   * @param[in] colind Column indices in CRS
   * @param[in] vals Data values in CRS
   * @param[in] b Input RHS
   * @param[in] nirs Number of IRs
   * @param[in] betas Length-two vector of residual bounds (low,high)
   * @param[out] x Output solution
   * @param[in] trans Transpose/Hermitian tag (optional, default false)
   * @param[in] r Final Schur complement rank (optional)
   */
  inline std::pair<size_t, int> hifir(
      const size_t n, const index_type *rowptr, const index_type *colind,
      const interface_type *vals, const interface_type *b, const size_t nirs,
      const double *betas, interface_type *x, const bool trans = false,
      const size_t r = static_cast<size_t>(-1)) const {
    constexpr static bool WRAP = true;

    const _array_type B(n, (_impl_type *)b, WRAP);
    _array_type       X(n, (_impl_type *)x, WRAP);
    _matrix_type      A(n, n, (index_type *)rowptr, (index_type *)colind,
                   (_impl_type *)vals, WRAP);
    return _base::hifir(A, B, nirs, betas, X, trans, r);
  }
};

/*!
 * @typedef di32PyHIF
 * @brief double-precision HIF
 */
typedef PyHIF<hif::HIF<double, int>> di32PyHIF;

/*!
 * @typedef di64PyHIF
 * @brief double-precision int64 HIF
 */
typedef PyHIF<hif::HIF<double, std::int64_t>> di64PyHIF;

/*!
 * @typedef si32PyHIF
 * @brief single-precision HIF
 */
typedef PyHIF<hif::HIF<float, int>> si32PyHIF;

/*!
 * @typedef si64PyHIF
 * @brief single-precision int64 HIF
 */
typedef PyHIF<hif::HIF<float, std::int64_t>> si64PyHIF;

/*!
 * @typedef zi32PyHIF
 * @brief double-precision complex HIF
 */
typedef PyHIF<hif::HIF<std::complex<double>, int>> zi32PyHIF;

/*!
 * @typedef zi64PyHIF
 * @brief double-precision int64 complex HIF
 */
typedef PyHIF<hif::HIF<std::complex<double>, std::int64_t>> zi64PyHIF;

/*!
 * @typedef ci32PyHIF
 * @brief single-precision complex HIF
 */
typedef PyHIF<hif::HIF<std::complex<float>, int>> ci32PyHIF;

/*!
 * @typedef ci64PyHIF
 * @brief single-precision int64 complex HIF
 */
typedef PyHIF<hif::HIF<std::complex<float>, std::int64_t>> ci64PyHIF;

}  // namespace hifir4py

#endif  // HIFIR4PY_HPP_
