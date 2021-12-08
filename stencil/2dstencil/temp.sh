export CUDA_VISIBLE_DEVICES=1
gettemp(){
	nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader
}
waittilltemp60c() {
  	max=60
  	let CUDA_VISIBLE_DEVICES=1-${CUDA_VISIBLE_DEVICES}
  	echo "now GPU is" ${CUDA_VISIBLE_DEVICES}
	while true; do
		val=$(gettemp)
		echo "tmp now" ${val}
		if [[ "$val"< "$max" ]]; then
			break
		fi 
		sleep 10
	done
}
