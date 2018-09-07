<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;
use libNeuralNetwork\RandomMatrix;
use libNeuralNetwork\DataFormatter;

class RNN
{
  protected $inputSize;
  protected $inputRange;
  protected $hiddenSizes;
  protected $outputSize;
  protected $learningRate;
  protected $decayRate;
  protected $smoothEps;
  protected $regc;
  protected $clipval;
  protected $json;
  protected $dataFormatter;
  
  protected $fnSetupData;
  protected $fnFormatDataIn;
  protected $fnFormatDataOut;
  
  protected $stepCache;
  protected $runs;
  protected $totalCost;
  protected $ratioClipped;
  protected $model;
  protected $initialLayerInputs;
  protected $inputLookup;
  protected $outputLookup;
  
  public static function fnDefaults()
  {
    $aResult = [
      'inputSize' => 20,
      'inputRange' => 20,
      'hiddenSizes' => [20,20],
      'outputSize' => 20,
      'learningRate' => 0.01,
      'decayRate' => 0.999,
      'smoothEps' => 1e-8,
      'regc' => 0.000001,
      'clipval' => 5,
      'json' => null,
      'dataFormatter' => null,        
      'fnSetupData' => function($aData) 
      {
        if (
          is_string($aData[0])
          && !is_array($aData[0])
          && (
            !isset($aData[0]['input'])
            || !isset($aData[0]['output'])
          )
        ) {
          return $aData;
        }
        
        $aValues = [];
        $aResult = [];
        
        if (is_string($aData[0]) || is_array($aData[0])) {
          if ($this->dataFormatter === null) {
            for ($iI = 0; $iI < count($aData); $iI++) {
              array_push($aValues, $aData[$iI]);
            }
            $this->dataFormatter = new DataFormatter($aValues);
          }
          for ($iI = 0, $iMax = count($aData); $iI < $iMax; $iI++) {
            $fnFormatDataIn = Closure::bind($this->fnFormatDataIn, $this);
            array_push($aResult, $fnFormatDataIn($aData[$iI]));
          }
        } else {
          if ($this->dataFormatter === null) {
            for ($iI = 0; $iI < count($aData); $iI++) {
              array_push($aValues, $aData[i]['input']);
              array_push($aValues, $aData[i]['output']);
            }
            $this->dataFormatter = DataFormatter::fromArrayInputOutput($aValues);
          }
          for ($iI = 0, $iMax = count($aData); $iI < $iMax; $iI++) {
            $fnFormatDataIn = Closure::bind($this->fnFormatDataIn, $this);
            array_push($aResult, $fnFormatDataIn($aData[$iI]['input'], $aData[$iI]['output']));
          }
        }
        
        return $aResult;
      },
      'fnFormatDataIn' => function($aInput, $aOutput = null) 
      {
        if ($this->dataFormatter) {
          if (isset($this->dataFormatter->indexTable['stop-input'])) {
            return $this->dataFormatter->toIndexesInputOutput($aInput, $aOutput);
          } else {
            return $this->dataFormatter->toIndexes($aInput);
          }
        }
        return $aInput;
      },
      'fnFormatDataOut' => function($aInput, $aOutput) 
      {
        if ($this->dataFormatter) {
          return join('', $this->dataFormatter->toCharacters($aOutput));
        }
        return $aOutput;
      },
    ];
  }
  
  public static function fnTrainDefaults()
  {
    return [
      'iterations' => 20000,
      'errorThresh' => 0.005,
      'log' => false,
      'logPeriod' => 10,
      'learningRate' => 0.3,
      'callback' => null,
      'callbackPeriod' => 10,
      'keepNetworkIntact' => false        
    ];    
  }
  
  function __construct($aOptions=[]) 
  {
    $aOptions = array_merge(self::fnDefaults(), $aOptions);

    foreach ($aOptions as $sKey => $mValue) {
      $this->$sKey = $mValue;
    }

    $this->stepCache = [];
    $this->runs = 0;
    $this->totalCost = null;
    $this->ratioClipped = null;
    $this->model = null;

    $this->initialLayerInputs = array_map(function() { new Matrix($this->hiddenSizes[0], 1); }, $this->hiddenSizes);
    $this->inputLookup = null;
    $this->outputLookup = null;
    
    $this->fnInitialize();
  }
  
  public function fnInitialize() 
  {
    $this->model = [
      'input' => null,
      'hiddenLayers' => [],
      'output' => null,
      'equations' => [],
      'allMatrices' => [],
      'equationConnections' => []
    ];

    if ($this->dataFormatter !== null) {
      $this->inputSize =
      $this->inputRange =
      $this->outputSize = count($this->dataFormatter->characters);
    }

    if ($this->json) {
      $this->fnFromJSON($this->json);
    } else {
      $this->fnMapModel();
    }
  }
  
  public function fnCreateHiddenLayers() 
  {
    //0 is end, so add 1 to offset
    array_push($this->model['hiddenLayers'], $this->fnGetModel($this->hiddenSizes[0], $this->inputSize));
    $iPrevSize = $this->hiddenSizes[0];

    for ($iD = 1; $iD < count($this->hiddenSizes); $iD++) { // loop over depths
      $iHiddenSize = $this->hiddenSizes[d];
      array_push($this->model['hiddenLayers'], $this->fnGetModel($iHiddenSize, $iPrevSize));
      $iPrevSize = $iHiddenSize;
    }
  }
  
  public function fnGetModel($iHiddenSize, $iPrevSize) 
  {
    return [
      //wxh
      'weight' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //whh
      'transition' => new RandomMatrix($iHiddenSize, $iPrevSize, 0.08),
      //bhh
      'bias' => new Matrix($iHiddenSize, 1)
    ];
  }
  
  public function fnGetEquation($oEquation, $oInputMatrix, $oPreviousResult, $aHiddenLayer) 
  {
    return $oEquation->fnRelu(
      $oEquation->fnAdd(
        $oEquation->fnAdd(
          $oEquation->fnMultiply(
            $aHiddenLayer['weight'],
            $oInputMatrix
          ),
          $oEquation->fnMultiply(
            $aHiddenLayer['transition'],
            $oPreviousResult
          )
        ),
        $aHiddenLayer['bias']
      )
    );
  }

  public function fnCreateInputMatrix() 
  {
    //0 is end, so add 1 to offset
    $this->model['input'] = new RandomMatrix($this->inputRange + 1, $this->inputSize, 0.08);
  }
  
  public function fnCreateOutputMatrix() 
  {
    //0 is end, so add 1 to offset
    //whd
    $this->model['outputConnector'] = new RandomMatrix(
      $this->outputSize + 1, 
      $this->hiddenSizes[$this->hiddenSizes.length - 1], 
      0.08
    );
    //0 is end, so add 1 to offset
    //bd
    $this->model['output'] = new Matrix($this->outputSize + 1, 1);
  }
  
  public function fnBindEquation() 
  {
    $oEquation = new Equation();
    $aOutputs = [];
    $aEquationConnection = count($this->model['equationConnections']) > 0
      ? $this->model['equationConnections'][count($this->model['equationConnections']) - 1]
      : $this->initialLayerInputs
      ;

    // 0 index
    $oOutput = $this->fnGetEquation(
      $oEquation, 
      $oEquation->fnInputMatrixToRow($this->model['input']),
      $aEquationConnection[0],
      $this->model['hiddenLayers'][0]
    );
    array_push($aOutputs, $oOutput);
    // 1+ indices
    for ($iI = 1, $iMax = count($this->hiddenSizes); $iI < $iMax; $iI++) {
      $oOutput = $this->fnGetEquation($oEquation, $oOutput, $aEquationConnection[i], $this->model['hiddenLayers'][$iI]);
      array_push($aOutputs, $oOutput);
    }

    array_push($this->model['equationConnections'], $aOutputs);
    $oEquation->fnAdd($oEquation->fnMultiply($this->model['outputConnector'], $oOutput), $this->model['output']);
    array_push($this->model['equations'], $oEquation);
  }
  
  public function fnMapModel() 
  {
    $this->fnCreateInputMatrix();
    if (!$this->model['input']) throw new Exception('net.model.input not set');
    array_push($this->model['allMatrices'], $this->model['input']);

    $this->fnCreateHiddenLayers();
    if (!count($this->model['hiddenLayers'])) throw new Exception('net.hiddenLayers not set');
    for ($iI = 0, $iMax = count($this->model['hiddenLayers']); $iI < $iMax; $iI++) {
      foreach ($this->model['hiddenLayers'][$iI] as $mProperty) {
        array_push($this->model['allMatrices'], $mProperty);
      }
    }

    $this->fnCreateOutputMatrix();
    if (!$this->model['outputConnector']) throw new Exception('net.model.outputConnector not set');
    if (!$this->model['output']) throw new Exception('net.model.output not set');

    array_push($this->model['allMatrices'], $this->model['outputConnector']);
    array_push($this->model['allMatrices'], $this->model['output']);
  }

  public function fnTrainPattern($aInput, $iLearningRate = null) 
  {
    $iError = $this->fnRunInput($aInput);
    $this->fnRunBackpropagate($aInput);
    $this->fnStep($iLearningRate);
    return $iError;
  }

  public function fnRunInput($aInput) 
  {
    $this->runs++;
    $iMax = count($aInput);
    $iLog2ppl = 0;
    $iCost = 0;
    $oEquation;
    
    while (count($this->model['equations']) <= count($aInput) + 1) {//last is zero
      $this->fnBindEquation();
    }
    for ($iInputIndex = -1, $iInputMax = count($aInput); $iInputIndex < $iInputMax; $iInputIndex++) {
      // start and end tokens are zeros
      $iEquationIndex = $iInputIndex + 1;
      $oEquation = $this->model['equations'][$iEquationIndex];

      $iSource = ($iInputIndex === -1 ? 0 : $aInput[$iInputIndex] + 1); // first step: start with START token
      $iTarget = ($iInputIndex === $iMax - 1 ? 0 : $aInput[$iInputIndex + 1] + 1); // last step: end with END token
      $oOutput = $oEquation->fnRun($iSource);
      // set gradients into log probabilities
      let logProbabilities = output; // interpret output as log probabilities
      let probabilities = softmax(output); // compute the softmax probabilities

      log2ppl += -Math.log2(probabilities.weights[target]); // accumulate base 2 log prob and do smoothing
      cost += -Math.log(probabilities.weights[target]);
      // write gradients into log probabilities
      logProbabilities.deltas = probabilities.weights.slice(0);
      logProbabilities.deltas[target] -= 1;
    }

    $this->totalCost = cost;
    return Math.pow(2, log2ppl / (max - 1));
  }
  
}
