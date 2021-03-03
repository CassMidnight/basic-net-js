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
  gradient: 0,
};

function createTemplateNode(){
  let newTemplate = {...nodeTemplate};
  Object.keys(newTemplate).forEach(prop => {
    if ( Array.isArray( newTemplate[prop] ) ){
      newTemplate[prop] = [];
    } else if ( typeof newTemplate[prop] === 'object') {
      throw "Error! Function createTemplateNode does not support deep copy!"
    }
  })
  return {...nodeTemplate, weights: [], newWeights: []}
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
  for(let layerIndex = 0; layerIndex < model.length; layerIndex++){
    for(let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){

      let inputLayerIndex = layerIndex - 1;

      //skip the input layer since it has no weights
      if (inputLayerIndex < 0){
        continue; 
      }
      //console.log(inputLayerIndex);
      let inputLayerLength = model[inputLayerIndex].length;

      //console.log('randomiseNodes', inputLayerIndex, inputLayerLength)

      for(let j = 0; j < inputLayerLength; j++ ){
        model[layerIndex][nodeIndex].weights.push(Math.random());
      }

      //console.log(model[layerIndex][nodeIndex].weights);

      // You may want to comment this back in later
      //model[layerIndex][nodeIndex].bias = Math.random();
    }
  }
  //logObject("randomised model", model);
}

function train(model, data, inputValueCount, outputValueCount){

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
  });

  //console.log(trainingData);

  //start with eta of 0.1
  let eta = 0.1;
  let presentations = 100;

  for (let k = 1; k <= presentations; k++){
    let sumPresentationError = 0;
    trainingData.forEach(exPair => {
      run(model, exPair);
      //console.log(exPair);
      calculateNodeDerivatives(model);
      let totalError = calculateError(model, exPair);
      //console.log(totalError)
      calculateNewWeights(model, eta, totalError);
      updateToNewWeights(model);
      //logObject("1 run",model)

      sumPresentationError += totalError;
    })
    console.log(k + "/" + presentations, sumPresentationError/trainingData.length);
  }

  console.log(model[model.length - 1].forEach(node => {
    console.log(node.value)
  }))

  niceLogModel(model)
}

function run(model, inputData){
  inputData.forEach((dataPoint, index) => {
    if (index < model[0].length){
      model[0][index].value = dataPoint;
    }
  })

  for (let i = 0; i < model.length - 1; i++){
    calculateLayers(model[i], model[i+1])
  }
}

function calculateLayers(layer1, layer2){
  for(nodeIndex = 0; nodeIndex < layer2.length; nodeIndex++){
    let node = layer2[nodeIndex];
    let sumOfWeights = node.weights.reduce((acc, weight, index) => {
      return acc + (weight * layer1[index].value);
    }, 0); 
    
    node.value = sigmoid(sumOfWeights + node.bias);
  };
}

function calculateNodeDerivatives(model){
  model.forEach(layer => {
    layer.forEach(node => {
      node.derivative = slope(sigmoid, node.value);
    }) 
  })
}

// Based vaguely on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
function calculateError(model, exPair){
  let expectedOutput = exPair.slice(4, 7);
  //console.log("calculateError", expectedOutput);
  let layerIndex = model.length - 1;

  let totalError = 0;
  let outputLayer = model[layerIndex];
  for (let nodeIndex = 0; nodeIndex < outputLayer.length; nodeIndex++){
    // 0 - 0.9 = -0.9
    
    // We want the square of the error
    //let error = (1/2) * Math.pow(expectedOutput[nodeIndex] - outputLayer[nodeIndex].value, 2);
    let error = Math.pow(expectedOutput[nodeIndex] - outputLayer[nodeIndex].value, 2);


    //console.log(error);
    outputLayer[nodeIndex].error = error;
    //outputLayer[nodeIndex].derivative = slope(sigmoid, error);
    //outputLayer[nodeIndex].derivative = slope(sigmoid, outputLayer[nodeIndex].value);


    // We need to total error of the layer for later
    totalError += outputLayer[nodeIndex].error;
  }

  return totalError;
}

function calculateNewWeights(model, eta, systemError){
  for (let layerIndex = model.length - 1; layerIndex >= 0; layerIndex--){
    //console.log("calculateNewWeights", "layerIndex",layerIndex)
    //console.log(`calculating weights for layer ${layerIndex}`);
    let layer = model[layerIndex];
    //logObject(`layer ${layerIndex} pre-change`, layer);
    layer.forEach((node, nodeIndex) => {
      //console.log("node index", nodeIndex)
      node.weights.forEach((weight, weightIndex) => {
        //console.log(nodeIndex, weightIndex)
        let newWeight = calcNewWeight(eta, systemError, model, layerIndex, nodeIndex, weightIndex);
        model[layerIndex][nodeIndex].newWeights[weightIndex] = newWeight;
      });
    }) 
    //logObject(`layer ${layerIndex} post-change`, layer);
  }
}

function updateToNewWeights(model){
  for (let layerIndex = model.length - 1; layerIndex >= 0; layerIndex--){
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
  dx = dx || 0.0000001;
  return (f(x+dx) - f(x)) / dx;
}

function calcNewWeight(eta, systemError, model, layerIndex, nodeIndex, weightIndex){
  //console.log(eta, systemError, layerIndex, nodeIndex, weightIndex);
  let oldWeight = model[layerIndex][nodeIndex].weights[weightIndex];

  // We need to limit this to ignore the first layer, not sure how tonight
  let inputFromWeight = model[layerIndex - 1][weightIndex].value;
  //console.log("inputFromWeight",inputFromWeight, layerIndex, nodeIndex, weightIndex)

  let systemGradient = calcSystemGradient(model, layerIndex, nodeIndex);
  //console.log("systemDerivitive", systemDerivitive)

  let delta = calcWeightDelta(systemError, inputFromWeight, eta, systemGradient)
  //console.log("delta", delta)

  //console.log("delta", delta);
  if (delta < 0){
    throw "Delta was negative";
  }

  return oldWeight + delta;
}

/**
 * This is the part dad means when he talks about the delta rule
 * @param {*} systemError 
 * @param {*} inputFromWeight 
 * @param {*} eta 
 * @param {*} systemGradient 
 */
function calcWeightDelta(systemError, inputFromWeight, eta, systemGradient){
  //console.log("calcWeightDelta", systemError, inputFromWeight, eta, systemGradient);
  //return systemError * inputFromWeight * eta * systemGradient;
  return inputFromWeight * eta * systemGradient;
}

function getNodeGradient(model, currentNodeLayer, currentNodeIndex){

}

/**
 * This is the part dad means when he talks about the chain rule
 */
function calcSystemGradient(model, currentNodeLayer, currentNodeIndex){
  let currentNode = model[currentNodeLayer][currentNodeIndex];
  //console.log(currentNode)

  
  // if its the output layer then just get the simple gradient
  if ( (model.length - 1) === currentNodeLayer){
    //this should probably not be set here
    currentNode.gradient = currentNode.error * currentNode.derivative;
    return currentNode.gradient; 
  }
  
  let systemGradient = 1;  // so we don't multiply by 0

  for (let layerIndex = model.length - 1 ; layerIndex > currentNodeLayer; layerIndex--){
    let layer = model[layerIndex];

    let layerGradient = 0;
    for (let nodeIndex = 0; nodeIndex < layer.length; nodeIndex++){
      let node = layer[nodeIndex];
      let nodeGradient = 0;
      if (layerIndex === model.length - 1){
        nodeGradient += node.gradient; // calculated inelegantly above
      } else {
        let nextLayer = model[layerIndex + 1];
        for (let weightIndex = 0; weightIndex < node.weights.length; weightIndex++){
          console.log("layer", layerIndex, "node", nodeIndex, "weight", weightIndex, "nextLayer", layerIndex+1)
          nodeGradient += (nextLayer[weightIndex].gradient * node.weights[weightIndex]);
        }
      }

      node.gradient = nodeGradient * node.derivative;

      layerGradient += node.gradient;
    }

    systemGradient *=  layerGradient;
  }

  return systemGradient * currentNode.derivative;
}

function logObject(label, obj){
  console.group(label);
  console.log(JSON.stringify(obj, null, 2));
  console.groupEnd(label);
}

function niceLogModel(model){
  console.group('model');
  model.forEach((layer, index) => {
    console.group(`layer ${index}`);
    console.log(index, JSON.stringify(layer, null, 2))
    console.group(`layer ${index}`);
  })
  console.groupEnd('model');
}