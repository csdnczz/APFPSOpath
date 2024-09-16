#! /bin/bash

for i in {1..30}
do
	python3 pso.py >> results.txt
done
