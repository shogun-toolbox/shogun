/**
 * This is a collection of useful macros taken from swig's source code with some
 * modifications so that they work with all languages. 
 */

// We have to gaurd each definition because macro definitions are not centralized in swig
// and actually are redefined multiple times in different places for multiple languages

#ifndef %str
  #define %str(Arg)           `Arg`
  #define %arg(X...) X  // wraps a macro argument (such as X<Y, Z>) safely into one argument
#endif

#ifndef SWIG_OK
  #define SWIG_OK                    (0)
  #define SWIG_ERROR                 (-1)
  #define SWIG_IsOK(r)               (r >= 0)
#endif

#ifndef %argument_fail
  #ifdef SWIG_fail
    #define %argument_fail(Arg...) SWIG_fail;
  #else
    #define %argument_fail(Arg...) (void *) 0;
  #endif
#endif

#ifndef %string_name
  #define %string_name(Name)                "SWIG_" %str(Name)
  #define %symbol_name(Name, Type...)       SWIG_ ## Name ## _ #@Type
#endif

#ifndef %fragment_name
  %include "typemaps/fragments.swg"
#endif

// To use these macros you have to define fragment names using the family
// of SWIG_AsVal*, SWIG_From* macros from "typemaps/fragments.swg"

%define %_val_in_typemap_with_args(Type,args...)
  %typemap(in, fragment=SWIG_AsVal_frag(Type)) Type (int ecode = 0) {
    Type& val = $1;
    ecode = SWIG_AsVal_dec(Type)(args, val);
    if (!SWIG_IsOK(ecode)) {
      %argument_fail(ecode, "$ltype", $symname, $argnum);
    }
  }
  %typemap(freearg) Type "";
  %typemap(in, fragment=SWIG_AsVal_frag(Type)) const Type & (Type temp, int ecode = 0) {  
    ecode = SWIG_AsVal_dec(Type)(args, temp);
    if (!SWIG_IsOK(ecode)) {
      %argument_fail(ecode, "$*ltype", $symname, $argnum);
      $1 = nullptr;
    } else {
      $1 = &temp;
    }
  }
  %typemap(freearg) const Type& "";
%enddef

#define %val_in_typemap_with_args(Type,args...) %_val_in_typemap_with_args(Type,%arg(args),$input)
#define %val_in_typemap(Type...) %_val_in_typemap_with_args(%arg(Type),$input)

%define %_val_out_typemap_with_args(Type,args...)
  %typemap(out, fragment=SWIG_From_frag(Type)) Type, const Type {
    $result = SWIG_From_dec(Type)(args $1);
  }
  %typemap(out, fragment=SWIG_From_frag(Type)) const Type& {
    $result = SWIG_From_dec(Type)(args *$1); 
  }
%enddef

#define %val_out_typemap_with_args(Type,args...) %_val_out_typemap_with_args(Type,%arg(args),)
#define %val_out_typemap(Type...) %_val_out_typemap_with_args(%arg(Type))
