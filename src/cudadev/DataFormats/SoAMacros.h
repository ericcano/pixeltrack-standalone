/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#ifndef DataStrcutures_SoAMacros_h
#define DataStrcutures_SoAMacros_h

#include <iostream>
#include <cassert>
#include "boost/preprocessor.hpp"
#include <Eigen/Core>

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_DEVICE_ONLY __device__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define SOA_HOST_ONLY
#define SOA_DEVICE_ONLY
#define SOA_HOST_DEVICE
#define SOA_HOST_DEVICE_INLINE inline
#endif

// compile-time sized SoA

// Helper template managing the value within it column
template<typename T>
class SoAValue {
public:
  SOA_HOST_DEVICE_INLINE SoAValue(size_t i, T * col): val_(col[i]) {}
  SOA_HOST_DEVICE_INLINE operator T&() { return val_; }
  SOA_HOST_DEVICE_INLINE operator const T&() const { return val_; }
  SOA_HOST_DEVICE_INLINE T& operator() () { return val_; }
  SOA_HOST_DEVICE_INLINE const T& operator() () const { return val_; }
  SOA_HOST_DEVICE_INLINE T* operator& () { return &val_; }
  SOA_HOST_DEVICE_INLINE const T* operator& () const { return &val_; }
  template <typename T2>
  SOA_HOST_DEVICE_INLINE T& operator= (const T2& v) { return val_ = v; }
  typedef T valueType;
  static constexpr auto valueSize = sizeof(T);
private:
  T &val_;
};

// Helper template managing the value within it column
template<class C>
class SoAEigenValue {
public:
  typedef C Type;
  typedef Eigen::Map<C, 0, Eigen::InnerStride<Eigen::Dynamic>> MapType;
  typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> CMapType;
  SOA_HOST_DEVICE_INLINE SoAEigenValue(size_t i, typename C::Scalar * col, size_t stride): 
    val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime,
              Eigen::InnerStride<Eigen::Dynamic>(stride)),
    crCol_(col),
    cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime,
              Eigen::InnerStride<Eigen::Dynamic>(stride)),
    stride_(stride) {}
  SOA_HOST_DEVICE_INLINE MapType& operator() () { return val_; }
  SOA_HOST_DEVICE_INLINE const CMapType& operator() () const { return cVal_; }
  SOA_HOST_DEVICE_INLINE operator C() { return val_; }
  SOA_HOST_DEVICE_INLINE operator const C() const { return cVal_; }
  SOA_HOST_DEVICE_INLINE C* operator& () { return &val_; }
  SOA_HOST_DEVICE_INLINE const C* operator& () const { return &cVal_; }
  template <class C2>
  SOA_HOST_DEVICE_INLINE MapType& operator= (const C2& v) { return val_ = v; }
  typedef typename C::Scalar ValueType;
  static constexpr auto valueSize = sizeof(C::Scalar);
  SOA_HOST_DEVICE_INLINE size_t stride() { return stride_; }
  template<typename OtherDerived>
  typename Eigen::MatrixBase<C>::template cross_product_return_type<OtherDerived>::type
  SOA_HOST_DEVICE_INLINE cross(const Eigen::MatrixBase<OtherDerived>& other) const { return cVal_.cross(other); }
  
  template<typename OtherType>
  typename Eigen::MatrixBase<C>::template cross_product_return_type<typename OtherType::MapType>::type
  SOA_HOST_DEVICE_INLINE cross(const OtherType& other) const { return cVal_.cross(other.cVal_); }
  
private:
  MapType val_;
  const typename C::Scalar * __restrict__ crCol_;
  CMapType cVal_;
  size_t stride_;
};

// Helper template to avoid commas in macro
template<class C>
struct EigenConstMapMaker {
  typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> Type;
  class DataHolder {
  public:
    DataHolder(const typename C::Scalar * data): data_(data) {}    
    EigenConstMapMaker::Type withStride(size_t stride) {
      return EigenConstMapMaker::Type(data_, C::RowsAtCompileTime, C::ColsAtCompileTime,
              Eigen::InnerStride<Eigen::Dynamic>(stride));
    }
  private:
    const typename C::Scalar * const data_;
  };
  static DataHolder withData(const typename C::Scalar * data) {
    return DataHolder(data);
  }
};

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2
#define _VALUE_TYPE_FUNDAMENTAL_COLUMN 3

#define SoA_scalar(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME)
#define SoA_column(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME)
#define SoA_eigenColumn(TYPE, NAME) (_VALUE_TYPE_EIGEN_COLUMN, TYPE, NAME)
#define SoA_FundamentalTypeColumn(TYPE, NAME) (_VALUE_TYPE_FUNDAMENTAL_COLUMN, TYPE, NAME)

/* Iterate on the macro MACRO and return the result as a comma separated list */
#define _ITERATE_ON_ALL_COMMA(MACRO, DATA, ...)                                                                                     \
  BOOST_PP_TUPLE_ENUM(                                                                                                              \
    BOOST_PP_SEQ_TO_TUPLE(                                                                                                          \
      _ITERATE_ON_ALL(MACRO, DATA, __VA_ARGS__)                                                                                     \
    )                                                                                                                               \
  )
/* Iterate MACRO on all elements */
#define _ITERATE_ON_ALL(MACRO, DATA, ...)                                                                                           \
  BOOST_PP_SEQ_FOR_EACH(MACRO, DATA,                                                                                                \
    BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)                                                                                           \
  )

/* Switch on macros depending on scalar / column type */
#define _SWITCH_ON_TYPE(VALUE_TYPE, IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN, IF_FUNDAMENTAL_COLUMN)                                   \
  BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_SCALAR),                                                                       \
    IF_SCALAR,                                                                                                                      \
    BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_COLUMN),                                                                     \
      IF_COLUMN,                                                                                                                    \
      BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_EIGEN_COLUMN),                                                             \
        IF_EIGEN_COLUMN,                                                                                                            \
        BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_FUNDAMENTAL_COLUMN),                                                           \
          IF_FUNDAMENTAL_COLUMN,                                                                                                    \
          BOOST_PP_EMPTY()                                                                                                          \
        )                                                                                                                           \
      )                                                                                                                             \
    )                                                                                                                               \
  )

/* dump SoA fields information; these should expand to, for columns:
 * Example:
 * declare_SoA_template(SoA,
 *   // predefined static scalars
 *   // size_t size;
 *   // size_t alignment;
 *
 *   // columns: one value per element
 *   SoA_FundamentalTypeColumn(double, x),
 *   SoA_FundamentalTypeColumn(double, y),
 *   SoA_FundamentalTypeColumn(double, z),
 *   SoA_eigenColumn(Eigen::Vector3d, a),
 *   SoA_eigenColumn(Eigen::Vector3d, b),
 *   SoA_eigenColumn(Eigen::Vector3d, r),
 *   SoA_column(uint16_t, colour),
 *   SoA_column(int32_t, value),
 *   SoA_column(double *, py),
 *   SoA_FundamentalTypeColumn(uint32_t, count),
 *   SoA_FundamentalTypeColumn(uint32_t, anotherCount),
 *
 *   // scalars: one value for the whole structure
 *   SoA_scalar(const char *, description),
 *   SoA_scalar(uint32_t, someNumber)
 * );
 * 
 * dumps as:
 * SoA(32, 64):
 *   sizeof(SoA): 152
 *  Column x_ at offset 0 has size 256 and padding 0
 *  Column y_ at offset 256 has size 256 and padding 0
 *  Column z_ at offset 512 has size 256 and padding 0
 *  Eigen value a_ at offset 768 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value b_ at offset 1536 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value r_ at offset 2304 has dimension (3 x 1) and per column size 256 and padding 0
 *  Column colour_ at offset 3072 has size 64 and padding 0
 *  Column value_ at offset 3136 has size 128 and padding 0
 *  Column py_ at offset 3264 has size 256 and padding 0
 *  Column count_ at offset 3520 has size 128 and padding 0
 *  Column anotherCount_ at offset 3648 has size 128 and padding 0
 *  Scalar description_ at offset 3776 has size 8 and padding 56
 *  Scalar someNumber_ at offset 3840 has size 4 and padding 60
 * Final offset = 3904 computeDataSize(...): 3904
 *
 */

#define _DECLARE_SOA_DUMP_INFO_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                     \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Dump scalar */                                                                                                               \
    std::cout << " Scalar " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                       \
              <<  " has size " << sizeof(CPP_TYPE) << " and padding "                                                               \
              << ((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment - sizeof(CPP_TYPE)                                    \
              <<  std::endl;                                                                                                        \
    offset+=((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment;                                                           \
  ,                                                                                                                                 \
    /* Dump column */                                                                                                               \
    std::cout << " Column " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                       \
              <<  " has size " << sizeof(CPP_TYPE) * nElements << " and padding "                                                   \
              << (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment - (sizeof(CPP_TYPE) * nElements)        \
              <<  std::endl;                                                                                                        \
    offset+=(((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                             \
  ,                                                                                                                                 \
    /* Dump Eigen column */                                                                                                         \
    std::cout << " Eigen value " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                  \
              <<  " has dimension (" << CPP_TYPE::RowsAtCompileTime << " x " << CPP_TYPE::ColsAtCompileTime  <<  ")"                \
              << " and per column size " << sizeof(CPP_TYPE::Scalar) * nElements << " and padding "                                 \
              << (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment                                 \
                    - (sizeof(CPP_TYPE::Scalar) * nElements)                                                                        \
              <<  std::endl;                                                                                                        \
    offset+=(((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment                                      \
              * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                          \
  ,                                                                                                                                 \
    /* Dump fundamental type column */                                                                                              \
    std::cout << " Column " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                       \
              <<  " has size " << sizeof(CPP_TYPE) * nElements << " and padding "                                                   \
              << (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment - (sizeof(CPP_TYPE) * nElements)        \
              <<  std::endl;                                                                                                        \
    offset+=(((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                             \
)

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME)                                                                                  \
  BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

/**
 * Computation of the column or scalar pointer location in the memory layout (at SoA construction time)
 */
#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                               \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE *>(curMem);                                                                   \
    curMem += (((sizeof(CPP_TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                                     \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE *>(curMem);                                                                   \
    curMem += (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                        \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar *>(curMem);                                                           \
    curMem += (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment_) + 1) * byteAlignment_                                 \
          * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                              \
    BOOST_PP_CAT(NAME, Stride_) = (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment_) + 1)                              \
          * byteAlignment_ / sizeof(CPP_TYPE::Scalar);                                                                              \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE *>(curMem);                                                                   \
    curMem += (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                        \
  )

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME)                                                                            \
  _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

/**
 * Computation of the column or scalar size for SoA size computation
 */
#define _ACCUMULATE_SOA_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                    \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    ret += (((sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                                          \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    ret += (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                              \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    ret += (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment                                       \
          * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                              \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    ret += (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                              \
  )

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME)                                                                                 \
  _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

/**
 * Value accessor of the const_element subclass.
 */
#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                        \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    CPP_TYPE const & NAME() { return soa_. NAME (); }                                                                               \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    CPP_TYPE const & NAME() { return * (soa_. NAME () + index_); }                                                                  \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    /* Ugly hack with a helper template to avoid having commas inside the macro parameter */                                        \
    EigenConstMapMaker<CPP_TYPE>::Type const NAME() {                                                                               \
      return EigenConstMapMaker<CPP_TYPE>::withData(soa_. NAME () + index_).withStride(soa_.BOOST_PP_CAT(NAME, Stride)());          \
    }                                                                                                                               \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    CPP_TYPE const & NAME() { return * (soa_. NAME () + index_ + 0 + 0); }                                                          \
  )

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                     \
  _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME

/**
 * Generator of parameters for (non-const) element subclass (expanded comma separated).
 */
#define _DECLARE_ELEMENT_VALUE_ARG_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                 \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    (CPP_TYPE *NAME)                                                                                                                \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    (CPP_TYPE::Scalar *NAME) (size_t BOOST_PP_CAT(NAME, Stride))                                                                    \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    (CPP_TYPE &NAME)                                                                                                                \
  )

#define _DECLARE_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                                                                              \
  _DECLARE_ELEMENT_VALUE_ARG_IMPL TYPE_NAME

/**
 * Generator of member initialization for constructor of element subclass
 */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL(VALUE_TYPE, CPP_TYPE, NAME, DATA)                                         \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    (NAME (DATA, NAME))                                                                                                             \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    (NAME (DATA, NAME, BOOST_PP_CAT(NAME, Stride)))                                                                                 \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    (NAME (NAME))                                                                                                                   \
  )

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME)                                                            \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_ELEMENT_VALUE_COPY_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    static_cast<CPP_TYPE &>(NAME) = static_cast<std::add_const<CPP_TYPE>::type &>(other.NAME);                                      \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    static_cast<CPP_TYPE>(NAME) = static_cast<std::add_const<CPP_TYPE>::type &>(other.NAME);                                        \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    NAME = static_cast<std::add_const<CPP_TYPE>::type &>(other.NAME);                                                               \
  )

#define _DECLARE_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_COPY_IMPL TYPE_NAME)

/**
 * Declaration of the members of the element subclass
 */
#define _DECLARE_ELEMENT_VALUE_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                              \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    SoAValue<CPP_TYPE> NAME;                                                                                                        \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    SoAEigenValue<CPP_TYPE> NAME;                                                                                                   \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    CPP_TYPE & NAME;                                                                                                                \
  )
    

#define _DECLARE_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME)                                                                           \
  _DECLARE_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME

/**
 * Parameters passed to element subclass constructor in operator[]
 */
#define _DECLARE_ELEMENT_CONSTR_CALL_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                               \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    (BOOST_PP_CAT(NAME, _))                                                                                                         \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    (BOOST_PP_CAT(NAME, _)) (BOOST_PP_CAT(NAME, Stride_))                                                                           \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    (BOOST_PP_CAT(NAME, _[index]))                                                                                                  \
  )

#define _DECLARE_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                                                                            \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME)

/**
 * Direct access to column pointer.
 */
#define _DECLARE_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                      \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    SOA_HOST_DEVICE_INLINE CPP_TYPE& NAME() { return * BOOST_PP_CAT(NAME, _); }                                                     \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    SOA_HOST_DEVICE_INLINE CPP_TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                                       \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    /* Unsupported for the moment TODO */                                                                                           \
    BOOST_PP_EMPTY()                                                                                                                \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    SOA_HOST_DEVICE_INLINE CPP_TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                                       \
  )

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                                                                   \
  BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

/**
 * Direct access to column pointer (const).
 */
#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    SOA_HOST_DEVICE_INLINE CPP_TYPE const& NAME() const { return * BOOST_PP_CAT(NAME, _); }                                         \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    SOA_HOST_DEVICE_INLINE CPP_TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                           \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    SOA_HOST_DEVICE_INLINE CPP_TYPE::Scalar const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                   \
    SOA_HOST_DEVICE_INLINE size_t BOOST_PP_CAT(NAME,Stride)() const { return BOOST_PP_CAT(NAME, Stride_); }                         \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    SOA_HOST_DEVICE_INLINE CPP_TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                           \
  )

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

/**
 * SoA class member declaration (column pointers).
 */
#define _DECLARE_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                                   \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                                       \
    /* Scalar */                                                                                                                    \
    CPP_TYPE * BOOST_PP_CAT(NAME, _);                                                                                               \
  ,                                                                                                                                 \
    /* Column */                                                                                                                    \
    CPP_TYPE * BOOST_PP_CAT(NAME, _);                                                                                               \
  ,                                                                                                                                 \
    /* Eigen column */                                                                                                              \
    CPP_TYPE::Scalar * BOOST_PP_CAT(NAME, _);                                                                                       \
    size_t BOOST_PP_CAT(NAME, Stride_);                                                                                             \
  ,                                                                                                                                 \
    /* Fundamental type column */                                                                                                   \
    CPP_TYPE * BOOST_PP_CAT(NAME, _);                                                                                               \
  )

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/**
 * Helper/friend class allowing SoA introspection.
 */
template <class C>
struct SoAIntrospector {
  friend C;
  SOA_HOST_DEVICE_INLINE size_t size() const { return parent_.nElements_; }
  SOA_HOST_DEVICE_INLINE size_t byteSize() const { return parent_.byteSize_; }
  SOA_HOST_DEVICE_INLINE size_t byteAlignment() const { return parent_.byteAlignment_; }
  SOA_HOST_DEVICE_INLINE std::byte* data() const { return parent_.mem_; }
  SOA_HOST_DEVICE_INLINE C cloneToNewAddress(std::byte* addr) { return C(addr, parent_.nElements_, parent_.byteAlignment_ ); }
  
private:
  SoAIntrospector(const C& parent): parent_(parent) {}
  const C& parent_;
};

/*
 * A macro defining a SoA (structure of variable sized arrays variant).
 */
#define declare_SoA_template(CLASS, ...)                                                                                            \
struct CLASS {                                                                                                                      \
                                                                                                                                    \
  /* these could be moved to an external type trait to free up the symbol names */                                                  \
  using self_type = CLASS;                                                                                                          \
                                                                                                                                    \
  /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                           \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                     \
   * up to compute capability 8.X.                                                                                                  \
   */                                                                                                                               \
  constexpr static size_t defaultAlignment = 128;                                                                                   \
                                                                                                                                    \
  /* dump the SoA internal structure */                                                                                             \
  SOA_HOST_ONLY                                                                                                                     \
  static void dump(size_t nElements, size_t byteAlignment = defaultAlignment) {                                                     \
    std::cout << #CLASS "(" << nElements << ", " << byteAlignment << "): " << std::endl;                                            \
    std::cout << "  sizeof(" #CLASS "): " << sizeof(CLASS) << std::endl;                                                            \
    size_t offset=0;                                                                                                                \
    _ITERATE_ON_ALL(_DECLARE_SOA_DUMP_INFO, ~, __VA_ARGS__)                                                                         \
    std::cout << "Final offset = " << offset << " computeDataSize(...): " << computeDataSize(nElements, byteAlignment) << std::endl;\
    std::cout << std::endl;                                                                                                         \
  }                                                                                                                                 \
  /* Helper function used by caller to externally allocate the storage */                                                           \
  static size_t computeDataSize(size_t nElements, size_t byteAlignment = defaultAlignment) {                                        \
    size_t ret = 0;                                                                                                                 \
    _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                                        \
    return ret;                                                                                                                     \
  }                                                                                                                                 \
                                                                                                                                    \
  friend SoAIntrospector<CLASS>;                                                                                                    \
  SOA_HOST_DEVICE_INLINE const SoAIntrospector<CLASS> soaMetadata() const { return  SoAIntrospector<CLASS>(*this); }                \
                                                                                                                                    \
/* Constructor relying on user provided storage */                                                                                  \
  SOA_HOST_ONLY CLASS(std::byte* mem, size_t nElements, size_t byteAlignment = defaultAlignment):                                   \
      mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
    auto curMem = mem_;                                                                                                             \
    _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                   \
    /* Sanity check: we should have reached the computed size, only on host code */                                                 \
    byteSize_ = computeDataSize(nElements_, byteAlignment_);                                                                        \
    if(mem_ + byteSize_ != curMem)                                                                                                  \
      throw std::out_of_range("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                                \
  }                                                                                                                                 \
                                                                                                                                    \
  /* Constructor relying on user provided storage */                                                                                \
  SOA_DEVICE_ONLY CLASS(bool devConstructor, std::byte* mem, size_t nElements, size_t byteAlignment = defaultAlignment):            \
      mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
    auto curMem = mem_;                                                                                                             \
    _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                   \
  }                                                                                                                                 \
                                                                                                                                    \
  /* AoS-like accessor to individual elements */                                                                                    \
  struct const_element {                                                                                                            \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    const_element(CLASS const& soa, int index) :                                                                                    \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                                            \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS const& soa_;                                                                                                              \
    const int index_;                                                                                                               \
  };                                                                                                                                \
                                                                                                                                    \
  struct element {                                                                                                                  \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    element(size_t index,                                                                                                           \
      /* Declare parameters */                                                                                                      \
      _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_VALUE_ARG, index, __VA_ARGS__)                                                         \
    ):                                                                                                                              \
      _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION, index, __VA_ARGS__)                                       \
       {}                                                                                                                           \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    element& operator=(const element& other) {                                                                                      \
      _ITERATE_ON_ALL(_DECLARE_ELEMENT_VALUE_COPY, ~, __VA_ARGS__)                                                                  \
      return *this;                                                                                                                 \
    }                                                                                                                               \
    _ITERATE_ON_ALL(_DECLARE_ELEMENT_VALUE_MEMBER, ~, __VA_ARGS__)                                                                  \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor */                                                                                                           \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  element operator[](size_t index) {                                                                                                \
    rangeCheck(index);                                                                                                              \
    return element(index,                                                                                                           \
        _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__) );                                                      \
  }                                                                                                                                 \
                                                                                                                                    \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  const element operator[](size_t index) const {                                                                                    \
    rangeCheck(index);                                                                                                              \
    return element(index,                                                                                                           \
        _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__) );                                                      \
  }                                                                                                                                 \
                                                                                                                                    \
  /* accessors */                                                                                                                   \
  _ITERATE_ON_ALL(_DECLARE_SOA_ACCESSOR, ~, __VA_ARGS__)                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ACCESSOR, ~, __VA_ARGS__)                                                                      \
                                                                                                                                    \
  /* dump the SoA internal structure */                                                                                             \
  template <typename T> SOA_HOST_ONLY friend void dump();                                                                           \
                                                                                                                                    \
private:                                                                                                                            \
  /* Range checker conditional to the macro _DO_RANGECHECK */                                                                       \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  void rangeCheck(size_t index) const {                                                                                             \
    if constexpr (_DO_RANGECHECK) {                                                                                                 \
      if (index >= nElements_) {                                                                                                    \
        printf("In " #CLASS "::rangeCheck(): index out of range: %zu with nElements: %zu\n", index, nElements_);                    \
        assert(false);                                                                                                              \
      }                                                                                                                             \
    }                                                                                                                               \
  }                                                                                                                                 \
                                                                                                                                    \
  /* data members */                                                                                                                \
  std::byte* mem_;                                                                                                                  \
  size_t nElements_;                                                                                                                \
  size_t byteSize_;                                                                                                                 \
  size_t byteAlignment_;                                                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                         \
}

#endif // ndef DataStrcutures_SoAMacros_h