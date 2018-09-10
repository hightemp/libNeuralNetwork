<?php

include '../src/Utilities.php';
include '../src/Lookup.php';
include '../src/DataFormatter.php';
include '../src/Matrix.php';
include '../src/RandomMatrix.php';
include '../src/Equation.php';
include '../src/RNN.php';
include '../src/LSTM.php';

error_reporting(E_ERROR);

$aTrainingData = [
  'Jane saw Doug.',
  'Doug saw Jane.',
  'Spot saw Doug and Jane looking at each other.',
  'It was love at first sight, and Spot had a frontrow seat. It was a very special moment for all.',
];

$oLSTM = new libNeuralNetwork\LSTM();
$aResult = $oLSTM->fnTrain($aTrainingData, [ 'iterations' => 1500 ]);
$sRun1 = $oLSTM->fnRun('Jane');
$sRun2 = $oLSTM->fnRun('Doug');
$sRun3 = $oLSTM->fnRun('Spot');
$sRun4 = $oLSTM->fnRun('It');

print_r($sRun1);
print_r($sRun2);
print_r($sRun3);
print_r($sRun4);
