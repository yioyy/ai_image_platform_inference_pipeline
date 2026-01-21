#!/bin/bash
for folder in $(ls ./|sort -r)
do
	if [ -f $folder ]
	then
		continue
	fi
	cd $folder
	find . -name "*.dcm"|parallel -I% --max-args 1 gdcmconv --raw % %
	cd ../
done
