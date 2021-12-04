# Spark Practical Work First Semester 2021/2022
## Group C27 - BIG DATA - UNIVERSIDAD POLITÉCNICA DE MADRID

### Group Members:
 * CATALAN GRIS LUCIA
 * RODRIGUEZ INSERTE PAU
 * MAROUF DANIEL

# About our Project

The objective of this work is to help students to put into practice the concepts learnt during the
theory lessons, and to get proficiency in the use of Spark and other related Big Data
technologies. In this exercise, the students are required to develop a Spark application that
creates a machine learning model for a real-world problem, using real-world data: Predicting the
arrival delay of commercial flights.


# Installation
We recommend Docker because it works on all operating systems.

## 1.  Docker
First install docker, [Docker Installation](https://docs.docker.com/engine/install/).
User Manual [Docs](https://docs.docker.com/desktop/windows/).


### Docker Build
Clone repository and build Docker: It takes around 3 minutes to build (the first time it may take longer as it needs to download some files).
```
git clone https://github.com/Maroufd/bigdataprojectgroup27
cd bigdataprojectgroup27
docker build -t spark_app .
```
**Run the docker **
```
docker run -v my-vol:/job testspark5:latest /job/main.py
```
