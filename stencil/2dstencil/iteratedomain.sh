

test_list=(512 768)
# a100_list=(512 768 1024 1536 2304 3072 4608 6912 9216 13824 18432 27648)
# v100_list=(512 1024 1280 2048 2560 4096 5120 8192 10240 16384 20480)
a100_list=(512 768 1024 1536 2304 3072 4608 6912 9216 13824 18432)
v100_list=(512 1024 1280 2048 2560 4096 5120 8192 10240 16384)



source ../temp.sh

# waittilltemp60c

# list_used=(${a100_list[@]})
list_used=(${test_list[@]})
if [ "${GPU}" = "v100" ] 
then
	list_used=(${v100_list[@]})
elif [ "${GPU}" = "a100" ]
then
	list_used=(${a100_list[@]})
else
	echo "usupport" ${GPU}
	exit 1
fi

rununit()
{
	for((i=0; i<${TEST}; i++))
	do
		# if [ "${GPU}" = "a100" ]
		# then
		# 	waittilltemp60c
		# 	./*baseline.exe $WIDTH $HEIGHT --iter=${ITER} --bdim=128 ${FLOATTYPE} ${CHECK} --async >> ./${FILE}
		# 	waittilltemp60c
		# 	./*baseline.exe $WIDTH $HEIGHT --iter=${ITER} --bdim=256 ${FLOATTYPE} ${CHECK} --async >> ./${FILE}
		# fi
		waittilltemp60c
		./*baseline.exe $WIDTH $HEIGHT --iter=${ITER} --bdim=128 ${FLOATTYPE} ${CHECK} --warmup>> ./${FILE}
		waittilltemp60c
		./*baseline.exe $WIDTH $HEIGHT --iter=${ITER} --bdim=256 ${FLOATTYPE} ${CHECK} --warmup>> ./${FILE}
	done
}
runtest()
{
	if((y!=x))
	then
		echo "now" $x $y
		echo "now" $y $x
		let WIDTH=${x}
		let HEIGHT=${y}
		rununit
		let WIDTH=${y}
		let HEIGHT=${x}
		rununit
		y=$x
	fi
	echo "now" $x $x
	let WIDTH=${x}
	let HEIGHT=${x}
	rununit
}

TEST=1

CHECK="--check"
ITER=3
FILE=domain_fp32_${GPU}_check.log
FLOATTYPE="--fp32"
y=${list_used[0]}
rm ${FILE}
for x in ${list_used[@]};
do
	runtest	
done 

FILE=domain_fp64_${GPU}_check.log
FLOATTYPE="--fp64"
y=${list_used[0]}
rm ${FILE}
for x in ${list_used[@]};
do
	runtest	
done 

#experiment would run twice
CHECK=""
ITER=1000
TEST=2


FILE=domain_fp32_${GPU}.log
FLOATTYPE="--fp32"
y=${list_used[0]}
rm ${FILE}
for x in ${list_used[@]};
do
	runtest	
done 


FILE=domain_fp64_${GPU}.log
FLOATTYPE="--fp64"
y=${list_used[0]}
rm ${FILE}
for x in ${list_used[@]};
do
	runtest	
done 
