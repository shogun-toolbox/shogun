def fix_extra_inheritance(r_file):
    # fixes extra inheritance/partial matching/vaccessors

    regexp = re.compile(", '_p_std__shared_ptrT_[^']*'\)")
    for index in range(len(r_file)):
        line = r_file[index]
        if line.find("setClass") != -1:
            line = regexp.sub(')', line)
        line = line.replace("pmatch(name, names(accessorFuns));",
                            "match(name, names(accessorFuns));")
        if line.find("vaccessors = c(") != -1:
            line = "vaccessors = NULL;\n"
        r_file[index] = line
    return r_file


def remove_finalizers(r_file):
    # remove finalizers from R code (they are only added to constructors)
    # we will add finalizers in the cpp code for all cases

    in_delete_function = False

    def condition(line):
        nonlocal in_delete_function
        # finalizer functions
        if line.find("# Start of delete_") != -1:
            in_delete_function = True
            return False
        if line.find("# Start of") != -1:
            in_delete_function = False
        if in_delete_function:
            return False
        # the line registering the finalizer in constructors
        if line.find("reg.finalizer") != -1:
            return False
        return True
    return list(filter(condition, r_file))


def add_finalizer_in_swig_type_info(cpp_file):
    # add finalizer field information in swig_type_info

    finalizer_line = ("  R_CFinalizer_t         finalizer;\t\t"
                      "/* function pointer to the destructor */\n")

    for index in range(len(cpp_file)):
        line = cpp_file[index]
        if line.startswith("typedef struct swig_type_info *"):
            cpp_file.insert(
                index + 1, "#include <Rinternals.h>\n#undef length;\n")
            index += 1
            continue
        if line.startswith("} swig_type_info;"):
            cpp_file.insert(index, finalizer_line)
            break
    return cpp_file


def change_delete_functions_signatures(cpp_file):
    # change signature of delete functions to be R_CFinalizer_t
    # specifically they should return void

    regexp = re.compile("return [^;]*;")
    in_delete_function = False
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        prev_line = cpp_file[index-1]
        if line.startswith("R_swig_delete_") and prev_line.startswith(
                "SWIGEXPORT SEXP"):
            cpp_file[index - 1] = "SWIGEXPORT void\n"
            in_delete_function = True
            continue
        if in_delete_function:
            cpp_file[index] = regexp.sub("return;", line).replace(
                "SWIG_exception(", "SWIG_Error(")
        if line == "}\n":
            in_delete_function = False
    return cpp_file


def remove_delete_from_callentry(cpp_file):
    # remove delete functions from call entry
    return list(filter(
        lambda line: line.find("{\"R_swig_delete_") == -1, cpp_file))


def make_delete_type_map(cpp_file):
    # creates a map from type name to its deletor/finalizer

    # assumes that change_delete_functions_signatures is run
    delete_type_map = dict()
    in_delete_function = False
    delete_fun_name = None
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        prev_line = cpp_file[index-1]
        if line.startswith("R_swig_delete_") and prev_line.startswith(
                "SWIGEXPORT void"):
            in_delete_function = True
            delete_fun_name = line[0:line.find('(')].strip()
            continue
        if not in_delete_function:
            continue
        if line.find("SWIG_exception_fail(SWIG_ArgError(") != -1:
            type_str = line.split("\"")[-4]
            delete_type_map[type_str] = delete_fun_name
        if line == "}\n":
            in_delete_function = False
    return delete_type_map


def add_finalizer_foreach_type(cpp_file):
    # add finalizer info for each type
    delete_type_map = make_delete_type_map(cpp_file)
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        if not line.startswith("static swig_type_info _swigt_"):
            continue
        type_str = line.split('"')[-2]
        for t in type_str.split("|"):
            if t not in delete_type_map:
                if t.startswith("std::shared_ptr< "):
                    t = t[len("std::shared_ptr< "):][: -len(" > *")] + " *"
            if t in delete_type_map:
                delete_fun = f'&{delete_type_map[t]}'
            else:
                delete_fun = "(R_CFinalizer_t) 0"
            cpp_file[index] = line[:-3] + ", " + delete_fun + "};\n"
    return cpp_file


def attach_finalizer_on_sexp_creation(cpp_file):
    in_newpointer_function = False
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        prev_line = cpp_file[index-1]
        if line.startswith("SWIG_R_NewPointerObj") and prev_line == \
                "SWIGRUNTIMEINLINE SEXP\n":
            in_newpointer_function = True
            continue
        if in_newpointer_function:
            if line.find("SET_S4_OBJECT(rptr);") != -1:
                cpp_file.insert(
                    index, ("  if (type->finalizer) R_RegisterCFinalizer("
                            "rptr, type->finalizer);\n"))
                break
    return cpp_file


def change_convertptr_macro(cpp_file):
    # replaces the macros defining SWIG_ConvertPtr and SWIG_ConvertPtrAndOwn
    # to use the new function defined in change_convertptr_function
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        if line.startswith("#define SWIG_ConvertPtr(obj, pptr, type, flags)"):
            cpp_file[index] = ("#define SWIG_ConvertPtr(obj, pptr, type, flags)"
                               "         "
                               "SWIG_R_ConvertPtrAndOwn(obj, pptr,type, flags, 0)\n")
            continue
        if line.startswith("#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)"):
            cpp_file[index] = ("#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)"
                               "SWIG_R_ConvertPtrAndOwn(obj, pptr, type, flags, own)\n")
            return cpp_file
    raise Exception(
        "Could not find SWIG_ConvertPtr/SWIG_ConvertPtrAndOwn macros")


def change_convertptr_function(cpp_file):
    # replaces SWIG_R_ConvertPtr function with a new version that handles
    # new memory.
    # also replaces SWIG_R_ConvertPtr function calls with SWIG_ConvertPtr

    new_function = """SWIG_R_ConvertPtrAndOwn(SEXP obj, void **ptr, swig_type_info *ty, int flags, int *own) {
  void *vptr;
  if (!obj)
    return SWIG_ERROR;
  if (obj == R_NilValue) {
    if (ptr)
      *ptr = NULL;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }

  vptr = R_ExternalPtrAddr(obj);
  if (ty) {
    swig_type_info *to =
        (swig_type_info *)R_ExternalPtrAddr(R_ExternalPtrTag(obj));
    if (to == ty) {
      if (ptr)
        *ptr = vptr;
    } else {
      swig_cast_info *tc = SWIG_TypeCheck(to->name, ty);
      int newmemory = 0;
      if (ptr) {
        *ptr = SWIG_TypeCast(tc, vptr, &newmemory);
        if (newmemory == SWIG_CAST_NEW_MEMORY) {
          assert(own);
          if (own)
            *own = *own | SWIG_CAST_NEW_MEMORY;
        }
      }
    }
  } else {
    if (ptr)
      *ptr = vptr;
  }
  return SWIG_OK;
}
"""
    in_convertptr_function = False
    function_begin = -1
    function_end = -1
    for index in range(len(cpp_file)):
        line = cpp_file[index]
        if line.startswith("SWIG_R_ConvertPtr(SEXP obj"):
            in_convertptr_function = True
            function_begin = index
            continue
        if in_convertptr_function and line == "}\n":
            function_end = index
            cpp_file = cpp_file[:function_begin] + \
                [new_function] + cpp_file[function_end+1:]
            break

    last_index = index
    for index in range(last_index, len(cpp_file)):
        line = cpp_file[index]
        cpp_file[index] = line.replace("SWIG_R_ConvertPtr", "SWIG_ConvertPtr")
    return cpp_file


if __name__ == '__main__':
    import re
    import sys

    if len(sys.argv) != 3:
        print("Usage: fix_generated_swig_files <swig R file> <swig cpp file>")

    print("Fixing generated swig files for the R interface...")

    with open(sys.argv[1], "r") as f:
        r_file = f.readlines()
    with open(sys.argv[2], "r") as f:
        cpp_file = f.readlines()

    r_mark = "# preprocessed\n"

    if r_file[0] != r_mark:
        r_file = fix_extra_inheritance(r_file)
        r_file = remove_finalizers(r_file)
        r_file.insert(0, r_mark)

        with open(sys.argv[1], "w") as f:
            f.writelines(r_file)

    cpp_mark = "/* preprocessed */\n"

    if cpp_file[0] != cpp_mark:
        cpp_file = add_finalizer_in_swig_type_info(cpp_file)
        cpp_file = change_delete_functions_signatures(cpp_file)
        cpp_file = remove_delete_from_callentry(cpp_file)
        cpp_file = add_finalizer_foreach_type(cpp_file)
        cpp_file = attach_finalizer_on_sexp_creation(cpp_file)
        cpp_file = change_convertptr_macro(cpp_file)
        cpp_file = change_convertptr_function(cpp_file)
        cpp_file.insert(0, cpp_mark)

        with open(sys.argv[2], "w") as f:
            f.writelines(cpp_file)
