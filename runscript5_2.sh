#!/usr/bin/bash

for d in ./RUNS/*
do
    python manage.py plot_del --run $d 
done



