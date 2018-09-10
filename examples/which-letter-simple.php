<?php

include '../src/Utilities.php';
include '../src/Lookup.php';
include '../src/DataFormatter.php';
include '../src/Matrix.php';
include '../src/RandomMatrix.php';
include '../src/Equation.php';
include '../src/NeuralNetwork.php';

error_reporting(E_ERROR);

function fnCharacter($sString) {
  return array_map(
    function ($v) {
      return (int) ('#' === $v);
    }, 
    str_split(trim($sString))
  );
}

$aA = fnCharacter(
  '.#####.' .
  '#.....#' .
  '#.....#' .
  '#######' .
  '#.....#' .
  '#.....#' .
  '#.....#'
);
$aB = fnCharacter(
  '######.' .
  '#.....#' .
  '#.....#' .
  '######.' .
  '#.....#' .
  '#.....#' .
  '######.'
);
$aC = fnCharacter(
  '#######' .
  '#......' .
  '#......' .
  '#......' .
  '#......' .
  '#......' .
  '#######'
);

$oNet = new libNeuralNetwork\NeuralNetwork([ "log" => true ]);
$oNet->fnTrain([
  [ 'input' => $aA, 'output' => [ 'a' => 1 ] ],
  [ 'input' => $aB, 'output' => [ 'b' => 1 ] ],
  [ 'input' => $aC, 'output' => [ 'c' => 1 ] ]
]);

$sResult = libNeuralNetwork\Utilities::fnLikely(
  fnCharacter(
    '.#####.' .
    '#.....#' .
    '#.....#' .
    '###.###' .
    '#.....#' .
    '#.....#' .
    '#.....#'
  ), 
  $oNet
);

var_dump($sResult);
