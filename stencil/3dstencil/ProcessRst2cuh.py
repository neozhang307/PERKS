import pandas as pd



# df_none = pd.read_csv('./star_80_256.log', names=['name', 'halo', 'arch', 'type', 'usesm', 'regfd','regs', 'sm', 'spilla', 'spillb'], header=None, delimiter=' ')
# nospill=df_none[(df_none['spilla']==0)&(df_none['spillb']==0)]
# d_result=nospill.groupby(["halo","arch","type","usesm"])["regfd"].max().reset_index()

# for row in d_result.iterrows():
# 	print('#define {}_{}_{}_{} {}'.format(row[1]["halo"],row[1]["arch"],row[1]["type"],row[1]["usesm"],row[1][4]))


def parsedata(shape,arch,register):
	df_none = pd.read_csv('./{}_{}_{}.log'.format(shape,arch,register), names=['name', 'halo', 'bdim', 'ipt','arch', 'type', 'usesm', 'regfd','regs', 'sm', 'spilla', 'spillb'], header=None, delimiter=' ')
	nospill=df_none[(df_none['spilla']==0)&(df_none['spillb']==0)]
	d_result=nospill.groupby(["halo","bdim", 'ipt',"arch","type","usesm"])["regfd"].max().reset_index()

	for row in d_result.iterrows():
		# print("#if HALO == {}".format(row[1]["halo"]))
		# print('#define {}_{}_{}_{}{} {}'.
			# format(shape,register,row[1]["arch"],row[1]["type"],"_usm" if row[1]["usesm"]==1 else "",row[1][4]))

		print("template<>")
		print("struct regfolder<{},{},{},{},{},{},{},{}>".
			format(row[1]["halo"], 
				"{}_shape".format(shape),
				row[1]["bdim"],
				row[1]["ipt"],
				register,
				int(arch)*10,
				"true" if row[1]["usesm"]==1 else "false",
				"float" if row[1]["type"]=="f" else "double"
				))
		print("{")
		print("\tstatic int const val = {};".format(row[1]["regfd"]))
		print("};")


shapes=["star","box","type0","poisson"]
archs=["70","80"]
registerlimits=[128,256]

# parsedata("star","80","256")

import sys
file = open("perksconfig.cuh", 'w')
sys.stdout = file
print("#pragma once")
print("#define star_shape (1)")
print("#define box_shape (2)")
print("#define type0_shape (3)")
print("#define poisson_shape (4)")
print("template<int halo, int shape, int bdim, int ipt, int registeramount, int arch,  bool useSM, class REAL>")
print("struct regfolder")
print("{")
print("\tstatic int const val = 0;")
print("};")
for larch in archs:
	for lshape in shapes:
		for lregister in registerlimits:
			parsedata(lshape,larch,lregister)