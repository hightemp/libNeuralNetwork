<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;

class Equation
{
  protected $inputRow;
  protected $inputValue;
  protected $states;
  
  function __construct()
  {
    $this->inputRow = 0;
    $this->inputValue = null;
    $this->states = [];
  }

  public function fnAdd($oLeft, $oRight)
  {
    if (count($oLeft->weights) !== count($oRight->weights)) {
      throw new Error('misaligned matrices');
    }
    $oProduct = new Matrix($oLeft->rows, $oLeft->columns);
    array_push($this->states, [
      'left' => $oLeft,
      'right' => $oRight,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnAdd',
      'backpropagationFn' => 'Matrix::fnAddB'
    ]);
    return $oProduct;
  }

  public function fnAllOnes($iRows, $iColumns) 
  {
    $oProduct = new Matrix($iRows, $iColumns);
    array_push($this->states, [
      'left' => $oProduct,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnAllOnes'
    ]);
    return $oProduct;
  }
  
  public function fnCloneNegative($oM) 
  {
    $oProduct = new Matrix($oM->rows, $oM->columns);
    array_push($this->states, [
      'left' => $oM,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnCloneNegative'
    ]);
    return product;
  }
  
  public function fnSubtract($oLeft, $oRight) 
  {
    if (count($oLeft->weights) !== count($oRight->weights)) {
      throw new Exception('misaligned matrices');
    }
    return $this->fnAdd(
      $this->fnAdd(
        $this->fnAllOnes($oLeft->rows, $oRight->columns), 
        $this->fnCloneNegative($oLeft)
      ), 
      $oRight
    );
  }
  
  public function fnMultiply($oLeft, $oRight)
  {
    if ($oLeft->columns !== $oRight->rows) {
      throw new Exception('misaligned matrices');
    }
    $oProduct = new Matrix($oLeft->rows, $oRight->columns);
    array_push($this->states, [
      'left' => $oLeft,
      'right' => $oRight,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnMultiply',
      'backpropagationFn' => 'Matrix::fnMultiplyB'
    ]);
    return $oProduct;
  }
  
  public function fnMultiplyElement($oLeft, $oRight)
  {
    if (count($oLeft->weights) !== count($oRight->weights)) {
      throw new Exception('misaligned matrices');
    }
    $oProduct = new Matrix($oLeft->rows, $oLeft->columns);
    array_push($this->states, [
      'left' => $oLeft,
      'right' => $oRight,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnMultiplyElement',
      'backpropagationFn' => 'Matrix::fnMultiplyElementB'
    ]);
    return $oProduct;
  }
  
  public function fnRelu($oM) 
  {
    $oProduct = new Matrix($oM->rows, $oM->columns);
    array_push($this->states, [
      'left' => $oM,
      'product' => $oProduct,
      'forwardFn' => 'Matrix::fnRelu',
      'backpropagationFn' => 'Matrix::fnReluB'
    ]);
    return $oProduct;
  }
  
  public function fnInput(&$oInput) 
  {
    $inputValue = $this->inputValue;
    array_push($this->states, [
      'product' => $oInput,
      'forwardFn' => function() use (&$oInput, $inputValue) {
        $oInput->weights = $inputValue;
      }
    ]);
    return $oInput;
  }
  
  public function fnInputMatrixToRow($oM) 
  {
    $oSelf = &$this;
    $oProduct = new Matrix($oM->columns, 1);
    array_push($this->states, [
      'left' => $oM,
      'right' => function() use (&$oSelf) {
        return $oSelf->inputRow;
      },
      'product' => $oProduct,
      'forwardFn' => 'Matrix::rowPluck',
      'backpropagationFn' => 'Matrix::rowPluckB'
    ]);
    return product;
  }

  public function fnSigmoid($oM) 
  {
    $oProduct = new Matrix($oM->rows, $oM->columns);
    array_push($this->states, [
      'left' => $oM,
      'product' => $oProduct,
      //'forwardFn' => sigmoid,
      //'backpropagationFn' => sigmoidB
    ]);
    return $oProduct;
  }
  
  public function fnTanh($oM) 
  {
    $oProduct = new Matrix($oM->rows, $oM->columns);
    array_push($this->states, [
      'left' => $oM,
      'product' => $oProduct,
      //'forwardFn' => tanh,
      //'backpropagationFn' => tanhB
    ]);
    return $oProduct;
  }
  
  public function fnObserve($oM) 
  {
    $iForward = 0;
    $iBackpropagate = 0;
    array_push($this->states, [
      'forwardFn' => function() use ($iForward) {
        $iForward++;
      },
      'backpropagationFn' => function() use ($iBackpropagate) {
        $iBackpropagate++;
      }
    ]);
    return $oM;
  }
  
  public function fnRun($iRowIndex = 0) 
  {
    $this->inputRow = $iRowIndex;
    $aState;
    for ($iI = 0, $iMax = count($this->states); $iI < $iMax; $iI++) {
      $aState = $this->states[$iI];
      if (!isset($aState['forwardFn'])) {
        continue;
      }
      $aState['forwardFn']($aState['product'], $aState['left'], $aState['right']);
    }

    return $aState['product'];
  }
  
  public function fnRunInput($iInputValue)  
  {
    $this->inputValue = $iInputValue;
    $aState;
    for ($iI = 0, $iMax = count($this->states); $iI < $iMax; $iI++) {
      $aState = $this->states[$iI];
      if (!isset($aState['forwardFn'])) {
        continue;
      }
      $aState['forwardFn']($aState['product'], $aState['left'], $aState['right']);
    }

    return $aState['product'];
  }
  
  public function fnRunBackpropagate($iRowIndex = 0)  
  {
    $this->inputRow = $iRowIndex;

    $iI = count($this->states);
    $aState;
    while ($iI-- > 0) {
      $aState = $this->states[$iI];
      if (!isset($aState['backpropagationFn'])) {
        continue;
      }
      $aState['backpropagationFn']($aState['product'], $aState['left'], $aState['right']);
    }

    return $aState['product'];
  }
}

