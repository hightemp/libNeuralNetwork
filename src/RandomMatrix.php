<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;
use libNeuralNetwork\Utilities;

class RandomMatrix extends Matrix
{
  protected $std;
  
  function __construct($iRows, $iColumns, $iStd) 
  {
    parent::__construct($iRows, $iColumns);
    $this->rows = $iRows;
    $this->columns = $iColumns;
    $this->std = $iStd;
    for($iI = 0, $iMax = count($this->weights); $iI < $iMax; $iI++) {
      $this->weights[$iI] = Utilities::fnRandomF(-$iStd, $iStd);
    }
    
  }
}