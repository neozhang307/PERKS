#!/bin/bash
readarray a< ./matrixurls.txt


for i in "${!a[@]}"
	do
		a_t=$(echo ${a[i]}|tr -d '\n')
		echo "download ${a_t}"
		wget ${a_t} 
	done