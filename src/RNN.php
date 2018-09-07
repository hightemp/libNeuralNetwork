<?php

namespace libNeuralNetwork;

use libNeuralNetwork\Matrix;
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
    $aHiddenSizes = $this->hiddenSizes;
    $aModel = $this->model;
    $aHiddenLayers = model['hiddenLayers'];
    //0 is end, so add 1 to offset
    array_push($aHiddenLayers, $this->fnGetModel($aHiddenSizes[0], $this->inputSize));
    $iPrevSize = $aHiddenSizes[0];

    for ($iD = 1; $iD < count($aHiddenSizes); $iD++) { // loop over depths
      $iHiddenSize = $aHiddenSizes[d];
      array_push($aHiddenLayers, $this->fnGetModel($iHiddenSize, $iPrevSize));
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
}
