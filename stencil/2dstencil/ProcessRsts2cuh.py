import pandas as pd



# df_none = pd.read_csv('./star_80_256.log', names=['name', 'halo', 'arch', 'type', 'usesm', 'regfd','regs', 'sm', 'spilla', 'spillb'], header=None, delimiter=' ')
# nospill=df_none[(df_none['spilla']==0)&(df_none['spillb']==0)]
# d_result=nospill.groupby(["halo","arch","type","usesm"])["regfd"].max().reset_index()

# for row in d_result.iterrows():
# 	print('#define {}_{}_{}_{} {}'.format(row[1]["halo"],row[1]["arch"],row[1]["type"],row[1]["usesm"],row[1][4]))


def parsedata(shape,arch,register):
	df_none = pd.read_csv('./{}_{}_{}.log'.format(shape,arch,register), names=['name', 'tile', 'halo', 'arch', 'type', 'usesm', 'regfd','regs', 'sm', 'spilla', 'spillb'], header=None, delimiter=' ')
	nospill=df_none[(df_none['spilla']==0)&(df_none['spillb']==0)]
	d_result=nospill.groupby(["halo","arch","type","usesm",'tile'])["regfd"].max().reset_index()

	for row in d_result.iterrows():
		# print("#if HALO == {}".format(row[1]["halo"]))
		# print('#define {}_{}_{}_{}{} {}'.
			# format(shape,register,row[1]["arch"],row[1]["type"],"_usm" if row[1]["usesm"]==1 else "",row[1][4]))

		print("template<>")
		print("struct regfolder<{},{},{},{},{},{},{}>".
			format(row[1]["halo"], 
				"true" if shape=="star" else "false",
				register,
				int(arch)*10,
				"true" if row[1]["usesm"]==1 else "false",
				"float" if row[1]["type"]=="f" else "double",
				row[1]["tile"]
				))
		print("{")
		print("\tstatic int const val = {};".format(row[1][4]))
		print("};")


shapes=["star","box"]
archs=["70","80"]
registerlimits=[128,256]

# parsedata("star","80","256")

import sys
file = open("perksconfig.cuh", 'w')
sys.stdout = file
print("#pragma once")
print("template<int halo, bool isstar, int registeramount, int arch,  bool useSM, class REAL, int tile=8>")
print("struct regfolder")
print("{")
print("\tstatic int const val = 0;")
print("};")
for larch in archs:
	for lshape in shapes:
		for lregister in registerlimits:
			parsedata(lshape,larch,lregister)