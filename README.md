# RaceTimeEstimator
Calculate running times for races and Ultras based on Machine Learning models from set of GPX

This is still on a Draft Version of a library that manages a set of GPX files and estimates a running time given a GPX with a course. The file is now only retrieving GPX standard data filled with Polar Heart Rate. This will be used for estimating some running efficiency metrics that can be used for feeding the model and use together with elevation data to estimate the running times.

The library will use different models based on:
- Riegel Formula to estimate race times
- Daniel Formula to consider elevation in the estimation
- Mean Times for track splits
- Regression Models for similar track splits.

This is a very draft model with an examaple program that could predict accurately short - medium races (up to 50Kms). It is intended to be completed and improved as soon as more data is available (GPX files).

This is being developed with an educational purpose only, and for private use with no warranty.
