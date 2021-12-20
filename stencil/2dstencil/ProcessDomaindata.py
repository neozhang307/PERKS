import pandas as pd
# df_none = pd.read_csv('./baseline_fp32_a100.log', names=['ptx', 'halo', 'async', 'widthx', 'widthy', 'iter','blk', 'grid', 'blkpsm', 'sm', 'ltc', 'speed'], header=None, delimiter='\t')

# rslt_df=df_none.reset_index()
# rslt_df = df_none[df_none['async']='0']



useasync=0
aimdblk=256
dataunit_fp64=8
dataunit_fp32=4
l2cache_a100=40
l2cache_v100=10
acuratethreashold=0.85

aiml2cache=l2cache_a100
aimdataunit=dataunit_fp32

def parsebaselinelog(filename,dataunit,l2cache):
	df_none = pd.read_csv(filename,  header=None, delimiter='\t')

	# print(rslt_df[rslt_df['2']=1])
	rslt_df = df_none.loc[(df_none[2]==useasync)&(df_none[6]==aimdblk)]
	pd.options.mode.chained_assignment = None  
	rslt_df["size"] = rslt_df.loc[:,3] * rslt_df.loc[:,4]*dataunit/1024/1024*2
	rslt_df = rslt_df.loc[rslt_df["size"]>l2cache
]
	peakperf = rslt_df[11].max()
	# print(peakperf)
	# print(dataunit)
	rslt_df = rslt_df.loc[rslt_df[11]>peakperf*acuratethreashold]
	# print(rslt_df.iloc[0])
	# rslt_df = rslt_df.iloc[3,4]
	return (rslt_df.iloc[0].at[3],rslt_df.iloc[0].at[4])

import sys

file = open("domain.sh", 'w')
sys.stdout = file
types=["fp32","fp64"]
archs=["a100","v100"]
for larch in archs:
	if larch=="a100":
		aiml2cache=40
	if larch=="v100":
		aiml2cache=10
	for ltype in types:
		if ltype=="fp32":
			aimdataunit=4
		if ltype=="fp64":
			aimdataunit=8

		newfilenname="./baseline_{}_{}.log".format(ltype,larch)
		# print(newfilenname)
		rst=parsebaselinelog(newfilenname,aimdataunit,aiml2cache)
		# print(rst)
		print("export {}_{}_x={}".format(ltype,larch,rst[0]))
		print("export {}_{}_y={}".format(ltype,larch,rst[1]))

# fp32_a100_filename='./baseline_fp32_a100.log'
# fp64_a100_filename='./baseline_fp64_a100.log'
# fp32_v100_filename='./baseline_fp32_v100.log'
# fp64_v100_filename='baseline_fp64_v100.log'

# fp32_a100=parsebaselinelog(fp32_a100_filename);
# fp64_a100=parsebaselinelog(fp64_a100_filename);
# fp32_v100=parsebaselinelog(fp32_v100_filename);
# fp64_v100=parsebaselinelog(fp64_v100_filename);
# print(fp64_v100)
# file = open(domain.sh, 'a')
# sys.stdout = file 
# print("export fp32_a100_x={}".format(fp32_a100[0]))
# print("export fp32_a100_y={}".format(fp32_a100[1]))