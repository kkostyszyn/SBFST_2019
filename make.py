f = open("tags.txt")
f = f.readlines()

m = open("lib\makefile", "w")

m.write("fst:\n")
for x in f:
    y = "\tfstcompile --isymbols=ins.txt --osymbols=outs.txt " + x[:-1] + ".att " + x[:-1] + ".fst\n"
    print(y)
    m.write(y)

m.write("\tmv *.fst lib_fst")
m.write("clean_att: \n\trm *.att\n")
m.write("clean_fst: \n\trm *.fst\n")

m.close()
