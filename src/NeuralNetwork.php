<?php

namespace libNeuralNetwork;

class NeuralNetwork 
{
  protected $hiddenSizes;
  protected $trainOpts;
  protected $sizes;
  protected $outputLayer;
  protected $biases;
  protected $weights;
  protected $outputs;
  protected $deltas;
  protected $changes;
  protected $errors;
  protected $errorCheckInterval;
  protected $activation;

  public static function fnTrainDefaults()
  {
    return [
      "iterations" => 20000,    // the maximum times to iterate the training data
      "errorThresh" => 0.005,   // the acceptable error percentage from training data
      "log" => false,           // true to use console.log, when a function is supplied it is used
      "logPeriod" => 10,        // iterations between logging out
      "learningRate" => 0.3,    // multiply's against the input and the delta then adds to momentum
      "momentum" => 0.1,        // multiply's against the specified "change" then adds to learning rate for change
      "callback" => null,       // a periodic call back that can be triggered while training
      "callbackPeriod" => 10,   // the number of iterations through the training data between callback calls
      "timeout" => 0            // the max number of milliseconds to train for
    ];
  }
  
  public static function fnDefaults() 
  {
    return [
      "binaryThresh" => 0.5,     // ¯\_(ツ)_/¯
      "hiddenLayers" => [3],     // array of ints for the sizes of the hidden layers in the network
      "activation" => 'sigmoid'  // Supported activation types ['sigmoid', 'relu', 'leaky-relu', 'tanh']
    ];
  }
  
  public static function fnValidateTrainingOptions($aOptions) 
  {
    $aValidations = [
      "iterations" => function ($v) { return is_numeric($v) && $v > 0; },
      "errorThresh" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "log" => function ($v) { return is_callable($v) || is_bool($v); },
      "logPeriod" => function ($v) { return is_numeric($v) && $v > 0; },
      "learningRate" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "momentum" => function ($v) { return is_numeric($v) && $v > 0 && $v < 1; },
      "callback" => function ($v) { return is_callable($v) || $v === null; },
      "callbackPeriod" => function ($v) { return is_numeric($v) && $v > 0; },
      "timeout" => function ($v) { return is_numeric($v) && $v > 0; }
    ];
    foreach (self::fnTrainDefaults() as $sKey => $mValue) {
      if (isset($aValidations[$sKey]) && !$aValidations[$sKey]($aOptions[$sKey])) {
        throw new Exception("[$sKey, {$aOptions[$sKey]}] is out of normal training range, your network will probably not train.");
      }
    }
  }
  
  function __construct($aOptions = []) 
  {
    //Object.assign(this, self::fnDefaults(), $aOptions);
    $this->hiddenSizes = $aOptions["hiddenLayers"];
    $this->trainOpts = [];
    $this->fnUpdateTrainingOptions(array_merge(self::fnTrainDefaults(), $aOptions));

    $this->sizes = null;
    $this->outputLayer = null;
    $this->biases = null; // weights for bias nodes
    $this->weights = null;
    $this->outputs = null;

    // state for training
    $this->deltas = null;
    $this->changes = null; // for momentum
    $this->errors = null;
    $this->errorCheckInterval = 1;
    
    //if (!$this->constructor.prototype.hasOwnProperty('runInput')) {
      $this->runInput = null;
    //}
    //if (!$this->constructor.prototype.hasOwnProperty('calculateDeltas')) {
      $this->calculateDeltas = null;
    //}
  }
  
  public function fnInitialize() 
  {
    if (!$this->sizes) 
      throw new Exception('Sizes must be set before initializing');

    $this->outputLayer = $this->sizes.length - 1;
    $this->biases = []; // weights for bias nodes
    $this->weights = [];
    $this->outputs = [];

    // state for training
    $this->deltas = [];
    $this->changes = []; // for momentum
    $this->errors = [];

    for ($iLayer = 0; $iLayer <= $this->outputLayer; $iLayer++) {
      $iSize = $this->sizes[$iLayer];
      $this->deltas[$iLayer] = zeros($iSize);
      $this->errors[$iLayer] = zeros($iSize);
      $this->outputs[$iLayer] = zeros($iSize);

      if ($iLayer > 0) {
        $this->biases[$iLayer] = randos($iSize);
        $this->weights[$iLayer] = array_fill(0, $iSize, 0);
        $this->changes[$iLayer] = array_fill(0, $iSize, 0);

        for ($iNode = 0; $iNode < $iSize; $iNode++) {
          $iPrevSize = $this->sizes[$iLayer - 1];
          $this->weights[$iLayer][$iNode] = randos($iPrevSize);
          $this->changes[$iLayer][$iNode] = zeros($iPrevSize);
        }
      }
    }

    $this->fnSetActivation();
  }

  public function fnSetActivation($sActivation=null) 
  {
    $this->activation = ($sActivation) ? $sActivation : $this->activation;
    switch ($this->activation) {
      case 'sigmoid':
        $this->runInput = $this->runInput || $this->_runInputSigmoid;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasSigmoid;
        break;
      case 'relu':
        $this->runInput = $this->runInput || $this->_runInputRelu;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasRelu;
        break;
      case 'leaky-relu':
        $this->runInput = $this->runInput || $this->_runInputLeakyRelu;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasLeakyRelu;
        break;
      case 'tanh':
        $this->runInput = $this->runInput || $this->_runInputTanh;
        $this->calculateDeltas = $this->calculateDeltas || $this->_calculateDeltasTanh;
        break;
      default:
        throw new Exception('unknown activation '.$this->activation.', The activation should be one of [\'sigmoid\', \'relu\', \'leaky-relu\', \'tanh\']');
    }
  }
  
  public function fnIsRunnable()
  {
    if(!$this->runInput){
      echo ('Activation function has not been initialized, did you run train()?');
      return false;
    }

    $checkFns = [
      'sizes',
      'outputLayer',
      'biases',
      'weights',
      'outputs',
      'deltas',
      'changes',
      'errors',
    ];
    
    $checkFns = array_filter($checkFns, function($v) { return $this->$v === null; });

    if(count($checkFns) > 0){
      echo ("Some settings have not been initialized correctly, did you run train()? Found issues with: ".join(",", $checkFns));
      return false;
    }
    return true;
  }
  
  public function fnRun($aInput) {
    //if (!$this->isRunnable) return null;
    if ($this->inputLookup) {
      $aInput = Lookup::fnToArray($this->inputLookup, $aInput);
    }

    $aOutput = $this->runInput($aInput);

    if ($this->outputLookup) {
      $aOutput = Lookup::fnToHash($this->outputLookup, $aOutput);
    }
    return $aOutput;
  }
}
