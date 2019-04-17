#!/usr/bin/env bash

/usr/bin/python3.6 /home/antonin/PycharmProjects/noise2noise/config.py --desc='SAR-NPY-3ch-Speckle-L1' train --train-tfrecords=datasets/mva-sar-npy-3ch.tfrecords --long-train=true --noise=speckle-l1 --val-dir=mva-sar-npy-3ch
/usr/bin/python3.6 /home/antonin/PycharmProjects/noise2noise/config.py validate --dataset-dir=datasets/mva-sar-npy-3ch/val --network-snapshot=results/00058-autoencoder'SAR-NPY-3ch-Speckle-L1'-n2n/network_final.pickle --noise=speckle-l1

/usr/bin/python3.6 /home/antonin/PycharmProjects/noise2noise/config.py --desc='SAR-NPY-3ch-Speckle-L5' train --train-tfrecords=datasets/mva-sar-npy-3ch.tfrecords --long-train=true --noise=speckle-l1 --val-dir=mva-sar-npy-3ch
/usr/bin/python3.6 /home/antonin/PycharmProjects/noise2noise/config.py validate --dataset-dir=datasets/mva-sar-npy-3ch/val --network-snapshot=results/00060-autoencoder'SAR-NPY-3ch-Speckle-L5'-n2n/network_final.pickle --noise=speckle-l5

/home/antonin/PycharmProjects/noise2noise/visualization.py
