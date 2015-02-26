PATH_TO_SELF=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
MODULAR_DIR=$PATH_TO_SELF/../../../src/interfaces/modular/
grep -Eor "(%rename\([^ ]*\))|(%template\([^ ]*\))" $MODULAR_DIR | grep -Eo "\([^ ]*\)" | grep -Eo "[^()]+" | grep -v "RealVector"