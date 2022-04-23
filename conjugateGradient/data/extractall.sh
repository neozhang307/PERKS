#!/bin/bash
for f in *.tar.gz; 
do 
	echo ${f}
	tar xzvf "$f"; 
done