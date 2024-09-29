const Network = require("./structures/network.js");
const mnist = require("mnist"); //Import prelabeled dataset
const express = require("express");
const app = express();
const { spawn } = require("child_process");

let set = mnist.set(8000, 2000); //Create a random training set of 8000 and test set of 2000
let trainingSet = set.training;
let testSet = set.test;

const model = new Network();

//https://stackoverflow.com/questions/23450534/how-to-call-a-python-function-from-node-js