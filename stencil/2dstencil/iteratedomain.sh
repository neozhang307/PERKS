

test_list=(512 768)
a100_list=(512 768 1024 1536 2304 3072 4608 6912 9216 13824 18432 27648)
v100_list=(512 1024 1280 2048 2560 4096 5120 8192 10240 16384 20480)


source ../temp.sh

# waittilltemp60c

# list_used=(${a100_list[@]})
list_used=(${test_list[@]})
if ((${GPU}="v100")) 
then
	list_used=(${v100_list[@]})
else
	list_used=(${a100_list[@]})
fi

rununit()
{
	waittilltemp60c
	./*baseline.exe $WIDTH $HEIGHT --iter=${ITER} --bdim=128 ${FLOATTYPE} ${CHECK} --async >> ${FILE}
	waittilltemp60c
	./*baseline* $WIDTH $HEIGHT --iter=${ITER} --bdim=128 ${FLOATTYPE} ${CHECK} >> ${FILE}
	waittilltemp60c
	./*baseline* $WIDTH $HEIGHT --iter=${ITER} --bdim=256 ${FLOATTYPE} ${CHECK} --async >> ${FILE}
	waittilltemp60c
	./*baseline* $WIDTH $HEIGHT --iter=${ITER} --bdim=256 ${FLOATTYPE} ${CHECK} >> ${FILE}
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

CHECK="--check"
ITER=3
FILE=baseline_fp32_${GPU}_check.log
FLOATTYPE="--fp32"
y=${list_used[0]}
for x in ${list_used[@]};
do
	runtest	
done 

FILE=baseline_fp64_${GPU}_check.log
FLOATTYPE="--fp64"
y=${list_used[0]}
for x in ${list_used[@]};
do
	runtest	
done 


CHECK=""
ITER=1000
FILE=baseline_fp32_${GPU}.log
FLOATTYPE="--fp32"
y=${list_used[0]}
for x in ${list_used[@]};
do
	runtest	
done 

FILE=baseline_fp64_${GPU}.log
FLOATTYPE="--fp64"
y=${list_used[0]}
for x in ${list_used[@]};
do
	runtest	
done 