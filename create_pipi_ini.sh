#!/bin/bash

L=24
T=48
lat=A40.24
path="/hiskp2/jost/correlationfunctions/pipi_chris/data/${lat}/3_gevp-data/"
#path="/hiskp2/correlators/${lat}/I2/"
path1=$(echo ${path} | sed -e 's/[\/&]/\\&/g')

echo TP0
#for ir in "A1g"; do
for ir in "A1g" "T1u" "T2g" "T2u" "Ep1g"; do
  sed -e "s/=LAT=/${lat}/;s/=PATH=/${path1}/;s/=P=/0/;s/=T=/${T}/;s/=L=/${L}/;s/=IR=/${ir}/" ini/pipi_template.ini > ini/pipi_TP0_${lat}_${ir}.ini
done

#echo TP1
#for ir in "A1g" "A1u" "A2g" "A2u" "Ep1g"; do
##for ir in "A1" "E2" "B1" "B2"; do
#  sed -e "s/=LAT=/${lat}/;s/=PATH=/${path1}/;s/=P=/1/;s/=T=/${T}/;s/=L=/${L}/;s/=IR=/${ir}/" ini/pipi_template.ini > ini/pipi_TP1_${lat}_${ir}.ini
#done
#
#echo TP2
#for ir in "A1g" "A1u" "A2g" "A2u"; do
##for ir in "A1" "A2" "B1" "B2"; do
#  sed -e "s/=LAT=/${lat}/;s/=PATH=/${path1}/;s/=P=/2/;s/=T=/${T}/;s/=L=/${L}/;s/=IR=/${ir}/" ini/pipi_template.ini > ini/pipi_TP2_${lat}_${ir}.ini
#done
#
#echo TP3
#for ir in "A1g" "A1u" "A2g" "A2u" "Ep1g"; do
##for ir in "A1" "A2" "E2"; do
#  sed -e "s/=LAT=/${lat}/;s/=PATH=/${path1}/;s/=P=/3/;s/=T=/${T}/;s/=L=/${L}/;s/=IR=/${ir}/" ini/pipi_template.ini > ini/pipi_TP3_${lat}_${ir}.ini
#done
