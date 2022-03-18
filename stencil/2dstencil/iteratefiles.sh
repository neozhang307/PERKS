
if [ -z "$GPU" ]
then
	echo "erro GPU not set"
	exit 1
fi
# echo "haha wrong"

for file in js2d5pt js2d9pt js2d13pt js2d17pt js2d21pt js2d25pt jb2d9pt jb2d25pt
# for file in  js2d5pt 
do
	cd ./${file}/
	echo "in" ${file}
	# bash ../iteratedomain_v100.sh
	# bash ../iteratemindomain.sh
	# cp domain.sh domain.sh.back
	# python ../ProcessDomaindata.py
	bash ../iteratebaseline_cm.sh
	echo "../iteratebaseline_cm.sh"
	bash ../iterategen.sh
	echo "../iterategen.sh"
	bash ../iterategenwr.sh
	echo "../iterategenwr.sh"
	cd ../
done
