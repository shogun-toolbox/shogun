from translate import Translator
import json
import unittest

class TestJavaTranslator(unittest.TestCase):

    def setUp(self):
        with open("targets/java.json", "r") as targetFile:
            self.translator = Translator(json.load(targetFile))

    def test_translateProgram(self):
        """
        CSVFile trainf("train.dat")
        RealFeatures feats_train(trainf)
        CSVFile testf("test.dat")

        Translates to:
        import org.shogun.*;
        import org.jblas.*;

        class Example {
            static {
                System.loadLibrary("modshogun");
            }

            public static void main(String argv[]) {
                modshogun.init_shogun_with_defaults();

                CSVFile trainf = new CSVFile("train.dat");
                RealFeatures feats_train = new RealFeatures(trainf);
                CSVFile testf = new CSVFile("test.dat");
            }
        }
        """
        programAST = [
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "trainf"},{"ArgumentList": {"Expr": {"StringLiteral": "train.dat"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "RealFeatures"}, {"Identifier": "feats_train"}, {"ArgumentList": {"Expr": {"Identifier": "trainf"}}}]}},
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "testf"}, {"ArgumentList": {"Expr": {"StringLiteral": "test.dat"}}}]}}
        ]

        translation = self.translator.translateProgram(programAST)

        self.assertEqual(translation, u"import org.shogun.*;\nimport org.jblas.*;\n\nclass Example {\n    static {\n        System.loadLibrary(\"modshogun\");\n    }\n\n    public static void main(String argv[]) {\n        modshogun.init_shogun_with_defaults();\n\n        CSVFile trainf = new CSVFile(\"train.dat\");\n        RealFeatures feats_train = new RealFeatures(trainf);\n        CSVFile testf = new CSVFile(\"test.dat\");\n\n    }\n}\n")

    def test_translateProgramWithNewlines(self):
        programAST = [
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "trainf"},{"ArgumentList": {"Expr": {"StringLiteral": "train.dat"}}}]}},
            {"Statement": "\n"},
            {"Statement": {"Init": [{"ObjectType": "RealFeatures"}, {"Identifier": "feats_train"}, {"ArgumentList": {"Expr": {"Identifier": "trainf"}}}]}},
            {"Statement": "\n"},
            {"Statement": {"Init": [{"ObjectType": "CSVFile"}, {"Identifier": "testf"}, {"ArgumentList": {"Expr": {"StringLiteral": "test.dat"}}}]}}
        ]

        translation = self.translator.translateProgram(programAST)

        self.assertEqual(translation, u"import org.shogun.*;\nimport org.jblas.*;\n\nclass Example {\n    static {\n        System.loadLibrary(\"modshogun\");\n    }\n\n    public static void main(String argv[]) {\n        modshogun.init_shogun_with_defaults();\n\n        CSVFile trainf = new CSVFile(\"train.dat\");\n\n        RealFeatures feats_train = new RealFeatures(trainf);\n\n        CSVFile testf = new CSVFile(\"test.dat\");\n\n    }\n}\n")

    def test_translateInitCopy(self):
        initAST = [
            {"ObjectType": "IntMatrix"},
            {"Identifier": "multiple_k"},
            {"Expr": {"MethodCall": [
                {"Identifier": "knn"},
                {"Identifier": "classify_for_multiple_k"}
            ]}}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"DoubleMatrix multiple_k = knn.classify_for_multiple_k()")

    def test_translateInitConstruct(self):
        initAST = [
            {"ObjectType": "MulticlassLabels"},
            {"Identifier": "labels"},
            {"ArgumentList": {
              "Expr": {"Identifier": "train_labels"}
            }}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"MulticlassLabels labels = new MulticlassLabels(train_labels)")

    def test_translateInitConstructMultiple(self):
        initAST = [
            {"ObjectType": "EuclideanDistance"},
            {"Identifier": "distance"},
            {"ArgumentList": [
              {"Expr": {"Identifier": "feats_train"}},
              {"Expr": {"Identifier": "feats_test"}}
            ]}
        ]
        translation = self.translator.translateInit(initAST)
        self.assertEqual(translation, u"EuclideanDistance distance = new EuclideanDistance(feats_train, feats_test)")

    def test_translateStatementAssign(self):
        stmtAST = {
            "Assign": [
                {"Identifier": "knn_train"},
                {"Expr":
                    {"BoolLiteral": "False"}
                }
            ]
        }
        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"        knn_train = false;\n")

    def test_translateStatementExpr(self):
        stmtAST = {
            "Expr": {
              "MethodCall": [
                {"Identifier": "knn"},
                {"Identifier": "train"}
              ]
            }
        }

        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"        knn.train();\n")

    def test_translateStatementNewLine(self):
        stmtAST = "\n"
        translation = self.translator.translateStatement(stmtAST)
        self.assertEqual(translation, u"\n")

    def test_translateStatementPrint(self):
        stmtAST = {
            "Print": {"Expr": {"Identifier": "multiple_k"}}
        }

        translation = self.translator.translateStatement(stmtAST)

        self.assertEqual(translation, u"        System.out.println(multiple_k);\n")

    def test_translateType(self):
        typeAST = {
            "ObjectType": "IntMatrix"
          }
        translation = self.translator.translateType(typeAST)

        self.assertEqual(translation, u"DoubleMatrix")

    def test_translateExprEnum(self):
        enumAST = {
            "Enum": [{"Identifier":"LIBLINEAR_SOLVER_TYPE"}, {"Identifier": "L2R_L2LOSS_SVC_DUAL"}]
        }
        translation = self.translator.translateExpr(enumAST)

        self.assertEqual(translation, u"L2R_L2LOSS_SVC_DUAL")
        self.assertTrue((u"LIBLINEAR_SOLVER_TYPE", u"L2R_L2LOSS_SVC_DUAL") in self.translator.dependencies["Enums"])

    def test_translateProgramComment(self):
        programAST = [
            {"Comment": " This is a comment"}
        ]
        translation = self.translator.translateProgram(programAST)

        trueTranslation = u"import org.shogun.*;\nimport org.jblas.*;\n\nclass Example {\n    static {\n        System.loadLibrary(\"modshogun\");\n    }\n\n    public static void main(String argv[]) {\n        modshogun.init_shogun_with_defaults();\n\n        // This is a comment\n\n    }\n}\n"
        self.assertEqual(translation, trueTranslation)


if __name__ == '__main__':
    unittest.main()