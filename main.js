fs = require('fs');

let nodeTemplate = {
  //Feed Forward
  value: 0.5,
  bias: 0,
  weights: [],

  //Back Propergation
  newWeights: [],
  error: 0,
  derivative: 0,
};

function createTemplateNode(){
  return {...nodeTemplate, weights: []}
}

let inputNodeCount = 4;
let outputNodeCount = 3;

let inputNodes = [];

for (i = 0; i < inputNodeCount; i++){
  inputNodes.push(createTemplateNode());
}

let layer1 = [];

for (i = 0; i < 2; i++){
  layer1.push(createTemplateNode());
}

//let layer2 = [];

//for (i = 0; i < 10; i++){
//  layer2.push(createTemplateNode());
//}

let outputNodes = [];

for (i = 0; i < outputNodeCount; i++){
  outputNodes.push(createTemplateNode());
}

//let model = [inputNodes, layer1, layer2, outputNodes];
let model = [inputNodes, layer1, outputNodes];


let irisData = [];

fs.readFile('irisTemp.csv', 'utf8', function (err,data) {
  let lines = data.split('\r\n');
  irisData = lines.map(line => {
    //0.5,0.33,0.14,0.02,1 
    let values = line.split(',');

    //console.log(values);

    let inputValues = values.slice(0, 4).map(val => parseFloat(val) ); 

    let outputValues = [];
    
    switch (values[4]){
      case '1': 
        outputValues = [1, 0, 0];
        break;
      case '2': 
        outputValues = [0, 1, 0];
        break;
      case '3': 
        outputValues = [0, 0, 1];
        break;
    } 

    return [ ...inputValues, ...outputValues ];
  });

  //console.log("irisData");
  //console.log(irisData);

  randomiseNodes(model);

  //console.log("model");
  //console.log(model);

  //run(model, [0.5, 0.5, 0.5, 0.5]);

  //train
  train(model, irisData);
});

function randomiseNodes(model){
  for(layerIndex = 0; layerIndex < model.length; layerIndex++){
    for(var nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){
      //console.log(layerIndex, nodeIndex);

      let inputLayerIndex = Math.max(layerIndex - 1, 0);
      //console.log(inputLayerIndex);
      let inputLayerLength = model[inputLayerIndex].length;

      //console.log('randomiseNodes', inputLayerIndex, inputLayerLength)

      for( j = 0; j < inputLayerLength; j++ ){
        model[layerIndex][nodeIndex].weights.push(Math.random());
      }

      //console.log(model[layerIndex][nodeIndex].weights);

      model[layerIndex][nodeIndex].bias = Math.random();
    }
  }
 
}

function train(model, data, inputValueCount, outputValueCount){
  //console.log(data);

  let thirds = Math.trunc(data.length / 3);

  let trainingData = [];
  let testingData = [];
  let validationData = [];

  data.forEach(exPair => {
    switch(random(1,4)){
      case 1: 
        trainingData.push(exPair)
        break;
      case 2: 
        testingData.push(exPair)
        break;
      case 3: 
        validationData.push(exPair)
        break;
    }
  })

  //start with eta of 0.1
  let eta = 0.1;
  let presentations = 3;

  for (let k = 0; k < presentations; k++){
    let sumPresentationError = 0;
    trainingData.forEach(exPair => {
      run(model, exPair);
      //logObject("model", model)
      let totalError = calculateError(model, exPair);
      //console.log("totalError", totalError);
      calculateNewWeights(model, eta, totalError);
      updatetoNewWeights(model);

      sumPresentationError += totalError;
    })
    console.log(k + "/" + presentations, sumPresentationError/trainingData.length);
  }

  console.log(model[model.length - 1].forEach(node => {
    console.log(node.value)
  }))

  logObject("model", model)
}

function run(model, inputData){
  inputData.forEach((dataPoint, index) => {
    if (index < model[0].length){
      model[0][index].value = dataPoint;
    }
  })

  for (let i = 0; i < model.length - 1; i++){
    calulateLayers(model[i], model[i+1])
  }
}

function calulateLayers(layer1, layer2){
  for(nodeIndex = 0; nodeIndex < layer2.length; nodeIndex++){
    let node = layer2[nodeIndex];
    let sumOfWeights = node.weights.reduce((acc, weight, index) => {
      //console.log(index)
      return acc + (weight * layer1[index].value);
    }, 0); 
    
    node.value = sigmoid(sumOfWeights + node.bias);
  };
}

// Based vagely on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
function calculateError(model, exPair){
  let expectedOutput = exPair.slice(4, 7);
  //console.log("calculateError", expectedOutput);
  let layerIndex = model.length - 1;

  let totalError = 0;
  let outputLayer = model[layerIndex];
  for (let nodeIndex = 0; nodeIndex < outputLayer.length; nodeIndex++){
    // 0 - 0.9 = -0.9
    
    // We want the square of the error
    let error = (1/2) * Math.pow(expectedOutput[nodeIndex] - outputLayer[nodeIndex].value, 2);

    //console.log(error);
    outputLayer[nodeIndex].error = error;
    outputLayer[nodeIndex].derivative = slope(sigmoid, error);

    // We need to total error of the layer for later
    totalError += outputLayer[nodeIndex].error;
  }

  return totalError;
}

function calculateNewWeights(model, eta, systemError){
  for (let layerIndex = model.length - 1; layerIndex > 0; layerIndex--){
    for (let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){
      for (let weightIndex = 0; weightIndex < model[layerIndex][nodeIndex].weights.length; weightIndex++){
        let newWeight = calcNewWeight(eta, systemError, model, layerIndex, nodeIndex, weightIndex);
        //console.log(newWeight);
        model[layerIndex][nodeIndex].newWeights[weightIndex] = newWeight;
      }
    }
  }
}

function updatetoNewWeights(model){
  for (let layerIndex = model.length - 1; layerIndex > 0; layerIndex--){
    for (let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){
      for (let weightIndex = 0; weightIndex < model[layerIndex][nodeIndex].weights.length; weightIndex++){
        model[layerIndex][nodeIndex].weights[weightIndex] = model[layerIndex][nodeIndex].newWeights[weightIndex];
      }
    }
  }
}

function sigmoid(x) {
  return 1 / ( 1 + Math.pow( Math.E, -1 * x) );
}

function random(min = 0, max = 100) {
  let num = Math.random() * (max - min) + min;
  return Math.floor(num);
};

function slope (f, x, dx) {
  dx = dx || .0000001;
  return (f(x+dx) - f(x)) / dx;
}

function calcNewWeight(eta, systemError, model, layerIndex, nodeIndex, weightIndex){
  //console.log(eta, systemError, layerIndex, nodeIndex, weightIndex);
  let oldWeight = model[layerIndex][nodeIndex].weights[weightIndex];

  // We need to limit this to ignore the first layer, not sure how tonight
  let inputFromWeight = model[layerIndex - 1][weightIndex].value;

  let systemDerivitive = calcSystemDerivitive(model, layerIndex, nodeIndex);
  //console.log("systemDerivitive", systemDerivitive)

  let delta = calcWeightDelta(systemError, inputFromWeight, eta, systemDerivitive)
  //console.log("delta", delta)

  return oldWeight + delta;
}

function calcWeightDelta(systemError, inputFromWeight, eta, systemDerivitive){
  //console.log("calcWeightDelta", systemError, inputFromWeight, eta, systemDerivitive);
  return systemError * inputFromWeight * eta * systemDerivitive;
}

function calcSystemDerivitive(model, currentNodeLayer, currentNodeIndex){
  let currentNode = model[currentNodeLayer][currentNodeIndex];
  //console.log(currentNode)

  let layerDerivitives = [];

  for (let i = model.length -1 ; i > currentNodeLayer; i--){
    let layer = model[i];

    let layerDerivitive = layer.reduce((layerAcc, node) => {
      return layerAcc + node.weights.reduce((nodeAcc, weight) => {
        return nodeAcc * (weight * node.derivative); 
      }, 1)
    }, 0)

    layerDerivitives.push(layerDerivitive);
  }

  return currentNode.derivative * layerDerivitives.reduce((acc, layerDerivitive) => acc * layerDerivitive, 1);
}

function logObject(label, obj){
  console.group(label);
  console.log(JSON.stringify(obj, null, 2));
  console.groupEnd(label);
}