import json
import sys
from string import Template
from sets import Set
import os.path
import argparse

class Translator:
    def __init__(self, targetDict):
        self.dependencies = {
            "AllClasses":Set(),
            "ConstructedClasses":Set(),
            "Enums":Set()
        }

        self.targetDict = targetDict

    def translateProgram(self, program, programName=None):
        """ Translate program AST
        Args:
            program: object like [statementAST, statementAST, statementAST, ...]
        """
        # reset dependencies
        self.dependencies["AllClasses"] = Set()
        self.dependencies["ConstructedClasses"] = Set()
        self.dependencies["Enums"] = Set()

        targetProgram = ""
        for line in program:
            if "Statement" in line:
                targetProgram += self.translateStatement(line["Statement"])
            elif "Comment" in line:
                targetProgram += self.translateComment(line["Comment"])

        programTemplate = Template(self.targetDict["Program"])

        return programTemplate.substitute(program=targetProgram, dependencies=self.dependenciesString(), programName=programName)

    def dependenciesString(self):
        """ Returns dependency import string 
            e.g. for python: "from modshogun import RealFeatures\n\n" 
        """

        if not "Dependencies" in self.targetDict:
            # Dependency strings are optional so we just return empty string 
            return ""

        # Three types of dependencies: a list of all classes used, 
        # a list of all explicitly constructed classes,
        # and list of all enums used. All are optional.
        allClassDependencies = ""
        constructedClassDependencies = ""
        enumDependencies = ""
        dependenciesExist = False

        if len(self.dependencies["AllClasses"]) > 0 and "AllClassDependencies" in self.targetDict["Dependencies"]:
            dependenciesExist = True
            template = Template(self.targetDict["Dependencies"]["AllClassDependencies"])
            allClassDependencies = template.substitute(classlist=self.seperatedClassDependencies("AllClasses"))

        if len(self.dependencies["ConstructedClasses"]) > 0 and "ConstructedClassDependencies" in self.targetDict["Dependencies"]:
            dependenciesExist = True
            template = Template(self.targetDict["Dependencies"]["ConstructedClassDependencies"])
            constructedClassDependencies = template.substitute(classlist=self.seperatedClassDependencies("ConstructedClasses"))

        if len(self.dependencies["Enums"]) > 0 and "EnumDependencies" in self.targetDict["Dependencies"]:
            dependenciesExist = True
            template = Template(self.targetDict["Dependencies"]["EnumDependencies"])
            enumDependencies = template.substitute(enums=self.seperatedEnumDependencies())

        if not dependenciesExist:
            return ""

        allDependenciesTemplate = Template(self.targetDict["Dependencies"]["AllDependencies"])
        return allDependenciesTemplate.substitute(allClassDependencies=allClassDependencies,
                                                  constructedClassDependencies=constructedClassDependencies,
                                                  enumDependencies=enumDependencies)

    def seperatedClassDependencies(self, type):
        if len(self.dependencies[type]) == 0:
            return ""

        dependencyList = list(self.dependencies[type])

        # Elements are just formatted as their names as default
        elementTemplate = Template("$element")
        if "DependencyListElementClass" in self.targetDict["Dependencies"]:
            elementTemplate = Template(self.targetDict["Dependencies"]["DependencyListElementClass"])

        # Retrieve list separator
        seperator = self.targetDict["Dependencies"]["DependencyListSeparator"]

        # separated dependencies
        csdependencies = ""
        for i, x in enumerate(dependencyList):
            csdependencies += elementTemplate.substitute(element=x)
            if i < len(dependencyList)-1:
                csdependencies += seperator

        return csdependencies

    def seperatedEnumDependencies(self):
        if len(self.dependencies["Enums"]) == 0:
            return ""

        dependencyList = list(self.dependencies["Enums"])

        # Enums are formatted as their value by default
        elementTemplate = Template("$value")
        if "DependencyListElementEnum" in self.targetDict["Dependencies"]:
            elementTemplate = Template(self.targetDict["Dependencies"]["DependencyListElementEnum"])

        # Retrieve list separator
        seperator = self.targetDict["Dependencies"]["DependencyListSeparator"]

        # separated dependencies
        sdependencies = ""
        for i, x in enumerate(dependencyList):
            sdependencies += elementTemplate.substitute(type=x[0],value=x[1])
            if i < len(dependencyList)-1:
                sdependencies += seperator

        return sdependencies


    def translateStatement(self, statement):
        """ Translate statement AST
        Args:
            statement: object like {"Init": initAST}, {"Assign": assignAST},
                       {"Expr": exprAST}, "\n"
        """
        if statement == "\n": # Newline handling
            return "\n"

        type = statement.keys()[0]
        translation = None
        if type == "Init":
            translation = self.translateInit(statement["Init"])
        elif type == "Assign":
            template = Template(self.targetDict["Assign"])
            name = statement["Assign"][0]["Identifier"]
            expr = self.translateExpr(statement["Assign"][1]["Expr"])
            translation = template.substitute(name=name, expr=expr)
        elif type == "Expr":
            translation = self.translateExpr(statement["Expr"])
        elif type == "Print":
            translation = self.translatePrint(statement["Print"])

        if translation == None:
            raise Exception("Unknown statement type: " + type)

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

        if initialisation.keys()[0] == "Expr":
            template = Template(self.targetDict["Init"]["Copy"])
            exprString = self.translateExpr(initialisation["Expr"])
            return template.substitute(name=nameString, type=typeString, expr=exprString)

        elif initialisation.keys()[0] == "ArgumentList":
            self.dependencies["ConstructedClasses"].add(typeString)
            template = Template(self.targetDict["Init"]["Construct"])
            argsString = self.translateArgumentList(initialisation["ArgumentList"])
            return template.substitute(name=nameString, type=typeString, arguments=argsString)

    def translateExpr(self, expr):
        """ Translate expression AST
        Args:
            expr: object like {"MethodCall": [identifierAST, identifierAST, argumentListAST]},
                  {"BoolLiteral": "False"}, {"StringLiteral": "train.dat"}, {"NumberLiteral": 4},
                  {"Identifier": "feats_test"}, etc.
        """
        key = expr.keys()[0]
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

            return template.substitute(object=object, method=method, arguments=translatedArgsList)

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
            # Add enum to dependencies in case they need to be imported explicitly
            self.dependencies["Enums"].add((expr[key][0]["Identifier"], expr[key][1]["Identifier"]))
            template = Template(self.targetDict["Expr"]["Enum"])
            return template.substitute(type=expr[key][0],value=expr[key][1]["Identifier"])

        raise Exception("Unknown expression type: " + key)

    def translatePrint(self, printStmt):
        template = Template(self.targetDict["Print"])
        return template.substitute(expr=self.translateExpr(printStmt["Expr"]))

    def translateComment(self, commentString):
        template = Template(self.targetDict["Comment"])
        return template.substitute(comment=commentString)

    def translateType(self, type):
        """ Translate type AST
        Args:
            type: object like {"ObjectType": "IntMatrix"}, {"BasicType": "float"}, etc.
        """
        template = ""
        typeKey = type.keys()[0]

        # Store dependency
        if typeKey == "ObjectType":
            self.dependencies["AllClasses"].add(type[typeKey])

        if type[typeKey] in self.targetDict["Type"]:
            template = Template(self.targetDict["Type"][type[typeKey]])
        else:
            template = Template(self.targetDict["Type"]["Default"])

        return template.substitute(type=type[typeKey])

    def translateArgumentList(self, argumentList):
        """ Translate argument list AST
        Args:
            argumentList: object like None, {"Expr": exprAST},
                          [{"Expr": exprAST}, {"Expr": exprAST}], etc.
        """
        if argumentList == None or argumentList == []:
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

def translate(ast, targetDict):
    translator = Translator(targetDict)
    programName = os.path.basename(ast["FilePath"]).split(".")[0]
    return translator.translateProgram(ast["Program"], programName)

def loadTargetDict(targetJsonPath):
    try:
        with open(targetJsonPath, "r") as jsonDictFile:
            return json.load(jsonDictFile)
    except IOError, err:
        if err.errno == 2:
            print "Target \"" + targetJsonPath + "\" does not exist"
        else:
            print err
        raise Exception("Could not load target dictionary")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", nargs='?', help="Translation target. Possible values: python, java, r, or octave. (default: python)")
    parser.add_argument("path", nargs='?', help="Path to input file. If not specified input is read from stdin")
    args = parser.parse_args()

    # Load target dictionary
    target = "python"
    if args.target:
        target = args.target
    targetDict = loadTargetDict("targets/" + target + ".json")

    # Read from input file (stdin or given path)
    programObject = None
    if args.path:
        with open(args.path, "r") as inputFile:
            programObject = json.load(inputFile)
    else:
        programObject = json.load(sys.stdin)

    print translate(programObject, targetDict)
