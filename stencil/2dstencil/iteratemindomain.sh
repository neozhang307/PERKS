
source ./domain.sh
source ../temp.sh


list_used=(${test_list[@]})
if [ "${GPU}" = "v100" ] 
then
	# list_used=(${v100_list[@]})
	echo "current GPU is " ${GPU}
elif [ "${GPU}" = "a100" ]
then
	echo "current GPU is " ${GPU}
else
	echo "usupport" ${GPU}
	exit 1
fi

rununit()
{
	for((i=0; i<${TEST}; i++))
	do
		waittilltemp60c
		./${executable} $WIDTH $HEIGHT --iter=${ITER} --bdim=${BDIM} --blkpsm=0 ${FLOATTYPE} ${CHECK} >> ./${FILE}
		waittilltemp60c
		./${executable} $WIDTH $HEIGHT --iter=${ITER} --bdim=${BDIM} --blkpsm=2 ${FLOATTYPE} ${CHECK} >> ./${FILE}
		waittilltemp60c
		./${executable} $WIDTH $HEIGHT --iter=${ITER} --bdim=${BDIM} --blkpsm=1 ${FLOATTYPE} ${CHECK} >> ./${FILE}
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
	
	else
		echo "now" $x $x
		let WIDTH=${x}
		let HEIGHT=${x}
		rununit
	fi

}

prefix=baseline

ITER=3
FLOATTYPE="--fp32"
y=${list_used[0]}

type=("fp32" "fp64")
# source ./domain.sh



FILEPREFIX="mindomain"
executable="*gen.exe"


for ltype in ${type[@]};
do
	FLOATTYPE="--${ltype}"
	FILE=${FILEPREFIX}_${ltype}_${GPU}.log
	./${executable} --checkmindomain ${FLOATTYPE} > ${FILE}
done 



