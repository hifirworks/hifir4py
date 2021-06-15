///////////////////////////////////////////////////////////////////////////////
//              This file is part of HIFIR4PY project                        //
//                                                                           //
//     Copyright (C) 2020 NumGeom Group at Stony Brook University            //
//                                                                           //
//     This program is free software: you can redistribute it and/or modify  //
//     it under the terms of the GNU Affero General Public License as        //
//     published by the Free Software Foundation, either version 3 of the    //
//     License, or (at your option) any later version.                       //
//                                                                           //
//     This program is distributed in the hope that it will be useful,       //
//     but WITHOUT ANY WARRANTY; without even the implied warranty of        //
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
//     GNU Affero General Public License  for more details.                  //
//                                                                           //
//     You should have received a copy of the GNU Affero General Public License //
//     along with this program.  If not, see <https://www.gnu.org/licenses/>.//
///////////////////////////////////////////////////////////////////////////////

// Authors:
//  Qiao,
// For enabling nullspace eliminator in Python via HIFIR4PY

#ifndef _HIFIR_PYTHON_HIFIR4PY_NSP_HPP
#define _HIFIR_PYTHON_HIFIR4PY_NSP_HPP

#include <Python.h>

#include <HIF.hpp>

namespace hif {

class PyNspFilter : public NspFilter {
 public:
  /// \brief default constructor
  /// \param[in] start (optional) starting position for constant null space
  /// \param[in] end (optional) ending position for constant null space
  explicit PyNspFilter(const std::size_t start = 0,
                       const std::size_t end   = static_cast<std::size_t>(-1))
      : NspFilter(start, end) {}

  virtual ~PyNspFilter() {
    if (user_call) Py_XDECREF(user_call);  // release the reference if we have
  }

  // encoder to numpy array, implemented in Cython
  using array_encoder_type         = PyObject *(*)(void *, std::size_t);
  array_encoder_type array_encoder = nullptr;

  // callback to invoke Python function, implemented in Cython
  using nsp_invoker_type       = void (*)(void *, PyObject *);
  nsp_invoker_type nsp_invoker = nullptr;

  PyObject *user_call = nullptr;  // user callback for customize the nullspace

  virtual void user_filter(void *x, const std::size_t n,
                           const char dtype) const override {
    if (dtype != 'd') hif_error("must be double precision");
    if (!array_encoder) hif_error("missing array encoder");
    if (!nsp_invoker) hif_error("missing nsp invoker");
    if (!user_call) hif_error("missing user callback");
    PyObject *ndarray = array_encoder(x, n);
    if (!ndarray) hif_error("failed to convert ndarray");
    nsp_invoker(user_call, ndarray);
    Py_XDECREF(ndarray);  // decrement the reference counter
  }

  // enable user override option
  inline void enable_or() { _type = NspFilter::USER_OR; }
};

}  // namespace hif

#endif  // _HIFIR_PYTHON_HIFIR4PY_NSP_HPP