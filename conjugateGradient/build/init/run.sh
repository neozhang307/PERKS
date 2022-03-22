
export PATHTODATA="../../data/"
readarray a< "${PATHTODATA}/filelist.txt"

export CUDA_VISIBLE_DEVICES=0

gettemp(){
   nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=temperature.gpu --format=csv,noheader
}
waittilltemp60c() {
   max=60

   while true; do
      val=$(gettemp)
      echo "tmp now" ${val}
      if [[ "$val"< "$max" ]]; then
         break
      fi 
      sleep 10
      let CUDA_VISIBLE_DEVICES=1-${CUDA_VISIBLE_DEVICES}
      echo "now GPU is" ${CUDA_VISIBLE_DEVICES}
   done
}

echo "gpu is" ${GPU}
export TEST=5

test(){
   rm cg_d_${GPU}.log
   for i in "${a[@]}"
   do
      echo "testing" "$i"
      # let BASIC=""
      for((iter=0; iter<${TEST}; iter++))
      do
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --mtx="${PATHTODATA}/$i" --baseline 2>>cg_d_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --mtx="${PATHTODATA}/$i" 2>>cg_d_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --mtx="${PATHTODATA}/$i" --cmat 2>>cg_d_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --mtx="${PATHTODATA}/$i" --cvec 2>>cg_d_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --mtx="${PATHTODATA}/$i" --cvec --cmat 2>>cg_d_${GPU}.log
      done
   done

   rm cg_${GPU}.log
   for i in "${a[@]}"
   do
      echo "testing" "$i"
      for((iter=0; iter<${TEST}; iter++))
      do
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --fp32 --baseline --mtx="${PATHTODATA}/$i" 2>>cg_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --fp32 --mtx="${PATHTODATA}/$i" 2>>cg_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --fp32 --cmat --mtx="${PATHTODATA}/$i" 2>>cg_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --fp32 --cvec --mtx="${PATHTODATA}/$i" 2>>cg_${GPU}.log
         waittilltemp60c
         ./cg_perks.exe --warmup --iters=10000 --staticiter --fp32 --cmat --cvec --mtx="${PATHTODATA}/$i" 2>>cg_${GPU}.log
      done
   done
}



test

# export PRE=coo
# test
# export PRE=col
# test
# export PRE=val
# test
# export PRE=vecr
# test
# export PRE=vecx
# test
# export PRE=vec
# test

# rm nbaseline_${GPU}.log

# for i in "${a[@]}"
# do
#    echo "testing" "$i"
#    for((iter=0; iter<${TEST}; iter++))
#    do
#       waittilltemp60c
#       ./conjugateGradientMultiBlockCG_baseline_nocoor --fp32 --mtx=$i 2>>nbaseline_${GPU}.log
#    done
# done

# rm nbaseline_d_${GPU}.log
# for i in "${a[@]}"
# do
#    echo "testing" "$i"
#    for((iter=0; iter<${TEST}; iter++))
#    do
#       waittilltemp60c
#       ./conjugateGradientMultiBlockCG_baseline_nocoor --mtx=$i 2>>nbaseline_d_${GPU}.log
#    done
# done

# echo "FINISH float"

# rm resultd_a100.log

# for i in "${a[@]}"
# do
#    echo "testing" "$i"
#    waittilltemp60c
#    ./conjugateGradientMultiBlockCG_spmv_d --mtx=$i 2>>resultd_a100.log
#    # break
#    waittilltemp60c
#    ./conjugateGradientMultiBlockCG_spmv_d --mtx=$i 2>>resultd_a100.log
#    # or do whatever with individual element of the array
# done