import json
import sys
from string import Template
import os.path
import argparse
import re
from copy import copy
from functools import reduce
# import set in a python 2->3 safe way
try:
    set
except NameError:
    from sets import Set as set


def find(key, dictionary):
    """ Recursively search a dictionary for a key """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def getDependencies(program):
    """ Traverses the program AST and extracts all dependencies """
    allClasses = set()
    interfaceClasses = set()
    enums = set()

    # All classes used
    for objectType in find("ObjectType", program):
        allClasses.add(objectType)

    for shogunSGType in find("ShogunSGType", program):
        allClasses.add(shogunSGType)

    # All classes where the constructor is called
    for initialisation in find("Init", program):
        # Constructor interface is used if an argument list is passed
        if list(initialisation[2].keys())[0] == "ArgumentList":
            typeDict = initialisation[0]
            objectKey = list(typeDict.keys())[0]
            interfaceClasses.add(typeDict[objectKey])

    # All classes on which a static method is called
    for staticCall in find("StaticCall", program):
        typeDict = staticCall[0]
        objectKey = list(typeDict.keys())[0]
        interfaceClasses.add(typeDict[objectKey])

    # All enums used
    for enum in find("Enum", program):
        enumType = enum[0]["Identifier"]
        enumValue = enum[1]["Identifier"]
        enums.add((enumType, enumValue))

    return allClasses, interfaceClasses, enums


def getVarsToStore(program):
    """ Extracts all variables in program that should be stored """
    varsToStore = []

    # store only real valued matrices, vectors and scalars
    for init in find("Init", program):
        typeObject = init[0]
        nameString = init[1]["Identifier"]

        append_store = False
        if "ShogunSGType" in init[0]:
            if init[0]["ShogunSGType"] in ("RealVector",
                                           "RealMatrix",
                                           "FloatVector",
                                           "FloatMatrix"):
                append_store = True

        elif "BasicType" in init[0]:
            if init[0]["BasicType"] in ("real", "float", "int"):
                append_store = True

        if append_store:
            varsToStore.append((typeObject, nameString))

    return varsToStore


class TranslationFailure(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Translator:
    def __init__(self, targetDict, tags={}):
        self.targetDict = targetDict
        self.tags = tags

    def translateProgram(self, program, programName=None, storeVars=False):
        """ Translate program AST
        Args:
            program: object like {"Program": [statementAST,
                                              statementAST,
                                              statementAST, ..]}
        """
        if storeVars:
            varsToStore = getVarsToStore(program)

            # Shallow copy of the statement list, so we don't alter the original
            program = copy(program)
            program["Program"] = copy(program["Program"])

            self.injectVarsStoring(program["Program"],
                                   programName,
                                   varsToStore)

        targetProgram = ""
        for line in program["Program"]:
            try:
                if "Statement" in line:
                    targetProgram += self.translateStatement(line["Statement"])
                elif "Comment" in line:
                    targetProgram += self.translateComment(line["Comment"])
            except Exception as e:
                print("Translation failed on line {}".format(line["__PARSER_INFO_LINE_NO"]))
                raise

        allClasses, interfacedClasses, enums = getDependencies(program)
        try:
            dependenciesString = self.dependenciesString(allClasses,
                                                         interfacedClasses,
                                                         enums)
        except Exception as e:
            print("Translation of dependencies failed!")
            raise

        programTemplate = Template(self.targetDict["Program"])
        return programTemplate.substitute(program=targetProgram,
                                          dependencies=dependenciesString,
                                          programName=programName)

    def injectVarsStoring(self, statementList, programName, varsToStore):
        """ Injects statements at the end of the program that perform variable
            storing
        """
        storage = "__sg_storage"
        storageFile = "__sg_storage_file"

        # TODO: handle directories
        storageFilename = {
            "Expr": {"StringLiteral": "{}.dat".format(programName)}
        }
        # 'w'
        storageFilemode = {"Expr": {"NumberLiteral": "119"}}
        storageComment = {"Comment": " Serialize output for integration testing (automatically generated)"}
        storageInit = {"Init": [{"ObjectType": "WrappedObjectArray"},
                                {"Identifier": storage},
                                {"ArgumentList": []}]}
        storageFileInit = {
            "Init": [{"ObjectType": "SerializableAsciiFile"},
                     {"Identifier": storageFile},
                     {"ArgumentList": [storageFilename, storageFilemode]}]
        }

        statementList.append({"Statement": "\n"})
        statementList.append(storageComment)
        statementList.append({"Statement": storageInit})
        statementList.append({"Statement": storageFileInit})

        for vartypeAST, varname in varsToStore:
            # avoid storing itself
            if varname in (storage, storageFile):
                continue

            varnameExpr = {"Expr": {"StringLiteral": varname}}
            varnameIdentifierExpr = {"Expr": {"Identifier": varname}}

            methodCall = {
                "MethodCall": [{"Identifier": storage},
                               {"Identifier": "append_wrapped"},
                               {"ArgumentList": [varnameIdentifierExpr,
                                                 varnameExpr]}]
            }
            expression = {"Expr": methodCall}
            statementList.append({"Statement": expression})

        storageSerialize = {
            "Expr": {"MethodCall": [
                {"Identifier": storage},
                {"Identifier": "save_serializable"},
                {"ArgumentList": [{"Expr": {"Identifier": storageFile}}]}
            ]}
        }
        statementList.append({"Statement": storageSerialize})

    def dependenciesString(self, allClasses, interfacedClasses, enums):
        """ Returns dependency import string
            e.g. for python: "from modshogun import RealFeatures\n\n"
        """

        if "Dependencies" not in self.targetDict:
            # Dependency strings are optional so we just return empty string
            return ""

        dependencies = set()

        if self.targetDict["Dependencies"].get("IncludeAllClasses"):
            dependencies = dependencies.union(allClasses)
        if self.targetDict["Dependencies"].get("IncludeInterfacedClasses"):
            dependencies = dependencies.union(interfacedClasses)
        if self.targetDict["Dependencies"].get("IncludeEnums"):
            dependencies = dependencies.union(enums)

        translations = set(map(self.translateDependencyElement, dependencies))

        separator = self.targetDict["Dependencies"]["DependencyListSeparator"]
        return reduce(lambda l, r: r if l == "" else l+separator+r,
                      translations,
                      "")

    def translateDependencyElement(self, dependencyElement):
        """ Translates a dependency element
        Args:
            dependencyElement: object like "RealFeatures"
                               or ("LIBLINEAR_SOLVER_TYPE", "L2R_L2LOSS_SVC")
        """
        dependencyRules = self.targetDict["Dependencies"]
        elementTemplate = Template(dependencyRules["DependencyListElement"])

        typeName = dependencyElement
        value = ""
        includePath = ""

        # If enum
        if isinstance(dependencyElement, tuple):
            typeName = dependencyElement[0]
            value = dependencyElement[1]

            if "DependencyListElementEnum" in dependencyRules:
                elementTemplate = Template(dependencyRules["DependencyListElementEnum"])

        elif "DependencyListElementClass" in dependencyRules:
            elementTemplate = Template(dependencyRules["DependencyListElementClass"])


        if "$includePath" in elementTemplate.template:
            includePath = self.getIncludePathForClass(typeName)

        return elementTemplate.substitute(typeName=typeName,
                                          value=value,
                                          includePath=includePath)

    def getIncludePathForClass(self, type_):
        translatedType = self.translateType({"ObjectType": type_})
        template_parameter_matcher = '\<[0-9a-zA-Z_]*\>'
        variants = [
            translatedType,
            'C' + translatedType,
            re.sub(template_parameter_matcher, '', translatedType),
            'C' + re.sub(template_parameter_matcher, '', translatedType)
        ]
        for variant in variants:
            if variant in self.tags:
                return self.tags[variant]

        raise TranslationFailure('Failed to obtain include path for %s' %
                                 (' or '.join(variants)))

    def translateStatement(self, statement):
        """ Translate statement AST
        Args:
            statement: object like {"Init": initAST}, {"Assign": assignAST},
                       {"Expr": exprAST}, "\n"
        """
        # Newline handling
        if statement == "\n":
            return "\n"

        # python2/3 safe dictionary keys
        type = list(statement.keys())[0]

        translation = None
        if type == "Init":
            translation = self.translateInit(statement["Init"])
        elif type == "Assign":
            translation = self.translateAssign(statement["Assign"])
        elif type == "Expr":
            translation = self.translateExpr(statement["Expr"])
        elif type == "Print":
            translation = self.translatePrint(statement["Print"])

        if translation is None:
            raise TranslationFailure("Unknown statement type: " + type)

        template = Template(self.targetDict["Statement"])
        return template.substitute(statement=translation)

    def translateInit(self, init):
        """ Translate initialisation statement AST
        Args:
            init: object like [typeAST, identifierAST, exprAST],
                  [typeAST, identifierAST, argumentListAST], etc.
        """
        typeString = self.translateType(init[0])
        nameString = init[1]["Identifier"]
        initialisation = init[2]
        typeKey = list(init[0].keys())[0]

        # python2/3 safe dictionary keys
        if list(initialisation.keys())[0] == "Expr":
            template = Template(self.targetDict["Init"]["Copy"])
            exprString = self.translateExpr(initialisation["Expr"])
            return template.substitute(name=nameString,
                                       typeName=typeString,
                                       expr=exprString)
        elif list(initialisation.keys())[0] == "ArgumentList":
            template = Template(self.targetDict["Init"]["Construct"])

            # Optional custom SGType construction
            if typeKey == "ShogunSGType"\
               and init[0][typeKey] in self.targetDict["Init"]:
                template = Template(self.targetDict["Init"][init[0][typeKey]])

            argsString = self.translateArgumentList(initialisation["ArgumentList"])
            return template.substitute(name=nameString,
                                       typeName=typeString,
                                       arguments=argsString)

    def translateAssign(self, assign):
        """ Translatie assignment AST
        Args:
            assign: object like [ElementAccessAST, expr] and [identifier, expr]
        """
        firstElementKey = list(assign[0].keys())[0]
        expr = assign[1]["Expr"]

        if firstElementKey == "Identifier":
            identifier = assign[0][firstElementKey]

            template = Template(self.targetDict["Assign"])
            exprString = self.translateExpr(expr)
            return template.substitute(identifier=identifier, expr=exprString)

        elif firstElementKey == "ElementAccess":
            return self.translateElementAccess(assign[0][firstElementKey],
                                               expr)

        else:
            raise TranslationFailure("Uknown assignment structure: " +
                                     str(assign))

    def translateExpr(self, expr):
        """ Translate expression AST
        Args:
            expr: objects like
                {"MethodCall": [identifierAST, identifierAST, argumentListAST]}
                {"BoolLiteral": "False"}
                {"StringLiteral": "train.dat"}
                {"NumberLiteral": 4}
                {"Identifier": "feats_test"}
                etc.
        """
        # python2/3 safe dictionary keys
        key = list(expr.keys())[0]

        if key == "MethodCall":
            template = Template(self.targetDict["Expr"]["MethodCall"])
            object = expr[key][0]["Identifier"]
            method = expr[key][1]["Identifier"]
            argsList = None
            try:
                argsList = expr[key][2]
            except IndexError:
                pass
            translatedArgsList = self.translateArgumentList(argsList)

            return template.substitute(object=object,
                                       method=method,
                                       arguments=translatedArgsList)

        elif key == "StaticCall":
            template = Template(self.targetDict["Expr"]["StaticCall"])
            type_ = self.translateType(expr[key][0])
            method = expr[key][1]["Identifier"]
            argsList = None
            try:
                argsList = expr[key][2]
            except IndexError:
                pass
            translatedArgsList = self.translateArgumentList(argsList)

            return template.substitute(typeName=type_,
                                       method=method,
                                       arguments=translatedArgsList)

        elif key == "ElementAccess":
            return self.translateElementAccess(expr[key])

        elif key == "BoolLiteral":
            return self.targetDict["Expr"]["BoolLiteral"][expr[key]]

        elif key == "StringLiteral":
            template = Template(self.targetDict["Expr"]["StringLiteral"])
            return template.substitute(literal=expr[key])

        elif key == "NumberLiteral":
            template = Template(self.targetDict["Expr"]["NumberLiteral"])
            return template.substitute(number=expr[key])

        elif key == "Identifier":
            template = Template(self.targetDict["Expr"]["Identifier"])
            return template.substitute(identifier=expr[key])

        elif key == "Enum":
            template = Template(self.targetDict["Expr"]["Enum"])
            return template.substitute(typeName=expr[key][0]["Identifier"],
                                       value=expr[key][1]["Identifier"])

        raise TranslationFailure("Unknown expression type: " + key)

    def translatePrint(self, printStmt):
        template = Template(self.targetDict["Print"])
        return template.substitute(expr=self.translateExpr(printStmt["Expr"]))

    def translateComment(self, commentString):
        template = Template(self.targetDict["Comment"])
        return template.substitute(comment=commentString)

    def translateType(self, type):
        """ Translate type AST
        Args:
            type: object like {"ObjectType": "IntMatrix"},
                              {"BasicType": "float"}, etc.
        """
        template = ""
        # python2/3 safe dictionary keys
        typeKey = list(type.keys())[0]

        if type[typeKey] in self.targetDict["Type"]:
            template = Template(self.targetDict["Type"][type[typeKey]])
        else:
            template = Template(self.targetDict["Type"]["Default"])

        return template.substitute(typeName=type[typeKey])

    def translateArgumentList(self, argumentList):
        """ Translate argument list AST
        Args:
            argumentList: object like None, {"Expr": exprAST},
                          [{"Expr": exprAST}, {"Expr": exprAST}], etc.
        """
        if argumentList is None or argumentList == []:
            return ""
        if isinstance(argumentList, list):
            head = argumentList[0]
            tail = argumentList[1:]

            translation = self.translateArgumentList(head)
            if len(tail) > 0:
                translation += ", " + self.translateArgumentList(tail)

            return translation
        else:
            if "Expr" in argumentList:
                return self.translateExpr(argumentList["Expr"])
            elif "ArgumentList" in argumentList:
                return self.translateArgumentList(argumentList["ArgumentList"])

    def translateElementAccess(self, elementAccess, expr=None):
        """ Translate element access AST
        Args:
            elementAccess: object like [identifierAST, argumentListAST]
            expr: exprAST - if given this method will use the element
                  assignment rule.
        """
        identifier = elementAccess[0]["Identifier"]
        indexList = elementAccess[1]["IndexList"]
        indexListTranslation = self.translateIndexList(indexList)

        exprString = None
        targetDict = self.targetDict["Element"]["Access"]
        if expr is not None:
            targetDict = self.targetDict["Element"]["Assign"]
            exprString = self.translateExpr(expr)

        template = None
        if len(indexList) == 1:
            template = Template(targetDict["Vector"])
        elif len(indexList) == 2:
            template = Template(targetDict["Matrix"])
        else:
            raise TranslationFailure("Element access takes either 1 index "
                                     "(vector) or 2 indices (matrix). Given "
                                     " " + str(len(indexList)) + " indices")

        return template.substitute(identifier=identifier,
                                   indices=indexListTranslation,
                                   expr=exprString)

    def translateIndexList(self, indexList):
        """ Translate index list AST
        Args:
            indexList: object like [NumberLiteralAST, NumberLiteralAST, ..]
        """
        addOne = not self.targetDict["Element"]["ZeroIndexed"]
        translation = ""
        for idx, numberLiteral in enumerate(indexList):
            try:
                index = int(numberLiteral["NumberLiteral"])
            except ValueError:
                raise TranslationFailure("Indices of element access must be "
                                         "integers.\n Error in literal: " +
                                         str(numberLiteral["NumberLiteral"]))

            if addOne:
                index += 1

            translation += str(index)

            if idx < len(indexList) - 1:
                translation += ", "

        return translation


def translate(ast, targetDict, tags, storeVars):
    translator = Translator(targetDict, tags)
    programName = os.path.basename(ast["FilePath"]).split(".")[0]

    return translator.translateProgram(ast,
                                       programName,
                                       storeVars)


def loadTargetDict(targetJsonPath):
    try:
        with open(targetJsonPath, "r") as jsonDictFile:
            return json.load(jsonDictFile)
    except IOError as err:
        if err.errno == 2:
            print("Target \"" + targetJsonPath + "\" does not exist")
        else:
            print(err)
        raise Exception("Could not load target dictionary")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        "--target",
                        nargs='?',
                        help="Translation target. Possible values: cpp, python, java, r, octave, csharp, ruby, lua. (default: python)")
    parser.add_argument("path",
                        nargs='?',
                        help="Path to input file. If not specified input is read from stdin")
    parser.add_argument("-g", "--ctags", help="path to ctags file")
    parser.add_argument("--store-vars",
                        help="whether to store all variables for testing",
                        action='store_true')
    args = parser.parse_args()

    # Load target dictionary
    target = "python"
    if args.target:
        target = args.target
    targetDict = loadTargetDict("targets/" + target + ".json")

    # Load ctags file
    tags = {}
    if args.ctags:
        from generate import parseCtags
        tags = parseCtags(args.ctags)

    storeVars = True if args.store_vars else False

    # Read from input file (stdin or given path)
    programObject = None
    if args.path:
        with open(args.path, "r") as inputFile:
            programObject = json.load(inputFile)
    else:
        programObject = json.load(sys.stdin)

    print(translate(programObject,
                    targetDict,
                    tags=tags,
                    storeVars=storeVars))
