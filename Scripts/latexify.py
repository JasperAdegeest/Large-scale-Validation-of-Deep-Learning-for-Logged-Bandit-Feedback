import numpy
import sys

#Expected Cmdline: python latexify.py [parser_log]
#Typical usage:
#python latexify.py sanitycheck.log

if len(sys.argv) < 2:
    print("Latexify:main \t [ERR] \t Expected Cmdline:  \
                python latexify.py [parser_log]",
                flush=True)
    sys.exit(0)

f = open(sys.argv[1], 'r')

slotID = None
approachIndex = None
resultLine = None

results = numpy.zeros((12, 7, 6), dtype = numpy.longdouble)

for line in f:
    if line.startswith("NumSlots"):
        slotID = int(line.replace("NumSlots:", '')) - 1
        resultLine = None
        approachIndex = None
    elif line.startswith("IPS"):
        approachIndex = 0
        resultLine = line[line.index('[')+2:]
    elif line.startswith("StdErr(IPS)"):
        approachIndex = 1
        resultLine = line[line.index('[')+2:]
    elif line.startswith("SN-IPS"):
        approachIndex = 2
        resultLine = line[line.index('[')+2:]
    elif line.startswith("AvgImpWt"):
        approachIndex = 3
        resultLine = line[line.index('[')+2:]
    elif line.startswith("StdErr(AvgImpWt)"):
        approachIndex = 4
        resultLine = line[line.index('[')+2:]
    elif line.startswith("BrokenImpWt"):
        approachIndex = 5
        resultLine = line[line.index('[')+2:]
    elif line.startswith("StdErr(BrokenImpWt)"):
        approachIndex = 6
        resultLine = line[line.index('[')+2:]
    elif ']' in line:
        if resultLine is not None:
            resultLine += line[:line.index(']')]
            resultLine = resultLine.replace('\n', ' ')

        if slotID is not None and approachIndex is not None:
            results[:, approachIndex, slotID] = [float(val) for val in resultLine.split()]
            approachIndex = None
            resultLine = None
    else:
        if resultLine is not None:
            resultLine += line

f.close()

for slotPairs in range(3):
    print("Slots " + str(2*slotPairs+1) + " and " + str(2*slotPairs+2), flush=True)
    print("$\\epsilon$ & ImpWt & IPS & SNIPS & ImpWt & IPS & SNIPS \\\\", flush=True)
    print("\\midrule", flush=True)
    for i in range(11, -1, -1):
        eps = None
        if i == 11:
            eps = "$0$"
        elif i == 0:
            eps = "$1$"
        else:
            eps = "$2^{-"+str(i)+"}$"

        ipsStr1 = "$" + ("%0.3f" % results[i, 0, 2*slotPairs]) + "\\! \\pm \\!" + ("%0.3f" % results[i, 1, 2*slotPairs]) + "$"
        snipsStr1 = "$" + ("%0.3f" % results[i, 2, 2*slotPairs]) + "$"
        impStr1 = "$" + ("%0.3f" % results[i, 3, 2*slotPairs]) + "\\! \\pm \\!" + ("%0.3f" % results[i, 4, 2*slotPairs]) + "$"

        ipsStr2 = "$" + ("%0.3f" % results[i, 0, 2*slotPairs + 1]) + "\\! \\pm \\!" + ("%0.3f" % results[i, 1, 2*slotPairs + 1]) + "$"
        snipsStr2 = "$" + ("%0.3f" % results[i, 2, 2*slotPairs + 1]) + "$"
        impStr2 = "$" + ("%0.3f" % results[i, 3, 2*slotPairs + 1]) + "\\! \\pm \\!" + ("%0.3f" % results[i, 4, 2*slotPairs + 1]) + "$"

        print(eps, "&", impStr1, "&", ipsStr1, "&", snipsStr1, "&", impStr2, "&", ipsStr2, "&", snipsStr2, "\\\\", flush=True)
    print("\\bottomrule", flush=True)
