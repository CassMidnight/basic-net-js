fs = require('fs');

let nodeTemplate = {
  //Feed Forward
  value: 0.5,
  bias: 0,
  weights: [],

  //Back Propagation
  error: 0,
  derivative: 0,
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

for (i = 0; i < 10; i++){
  layer1.push(createTemplateNode());
}

let layer2 = [];

for (i = 0; i < 10; i++){
  layer2.push(createTemplateNode());
}

let outputNodes = [];

for (i = 0; i < outputNodeCount; i++){
  outputNodes.push(createTemplateNode());
}

let model = [inputNodes, layer1, layer2, outputNodes];
//let model = [inputNodes, layer1, outputNodes];


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
  //randomiseNodes(model, true);

  //train
  train(model, irisData);
});

function randomiseNodes(model, debugMode = false){
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
        if (debugMode){
          // Make the generation predictable for testing
          model[layerIndex][nodeIndex].weights.push(0.1 * (nodeIndex + layerIndex + j));
        } else {
          model[layerIndex][nodeIndex].weights.push(Math.random());
        }
        
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

  //start with eta of 0.1
  let eta = 0.1;
  let presentations = 1000;

  for (let k = 1; k <= presentations; k++){
    let sumPresentationError = 0;
    trainingData.forEach(exPair => {
      run(model, exPair);

      let totalError = calculateError(model, exPair);

      // ----- Output Layer -----
      const outputLayerIndex = model.length - 1;
      for (let outputNodeIndex = 0; outputNodeIndex < model[outputLayerIndex].length; outputNodeIndex++){
        let node = model[outputLayerIndex][outputNodeIndex];
        // The comment below may be correct, check with dad
        //let partialDerivativeOfNode = node.value * slope(sigmoid, node.value) * node.error;
        let partialDerivativeOfNode = node.value * (1 - node.value) * node.error;
        //console.log(partialDerivativeOfNode, node.value, (1 - node.value), node.error);
        node.derivative = partialDerivativeOfNode;
        
        for (let weightIndex = 0; weightIndex < node.weights.length; weightIndex++){
          let weightDelta = eta * model[outputLayerIndex - 1][weightIndex].value * partialDerivativeOfNode;
          node.weights[weightIndex] += weightDelta;
        }
      }

      // ----- Hidden Layers ----- //
      // layerIndex > 0 ignores the first layers since we don't care about the input layer
      for (let layerIndex = outputLayerIndex - 1; layerIndex > 0; layerIndex--){
        for (let nodeIndex = 0; nodeIndex < model[layerIndex].length; nodeIndex++){
          let node = model[layerIndex][nodeIndex];

          for (let weightIndex = 0; weightIndex < node.weights.length; weightIndex++){
            //console.log(layerIndex, nodeIndex, weightIndex)
       
            let sumOfForwardLayersNodesPartialDerivativesAndWeights = model[layerIndex + 1].reduce((acc, forwardNode) => {
              //console.log(forwardNode.derivative, forwardNode.weights[nodeIndex]);
              return acc + (forwardNode.derivative * forwardNode.weights[nodeIndex]);
            },0);

            // The comment below may be correct, check with dad
            //let partialDerivativeOfNode = 0 * slope(sigmoid, node.value) * sumOfForwardLayersNodesPartialDerivativesAndWeights;
            let partialDerivativeOfNode =  node.value * (1 - node.value) * sumOfForwardLayersNodesPartialDerivativesAndWeights;
            node.derivative = partialDerivativeOfNode;

            let weightDelta = eta * model[layerIndex - 1][weightIndex].value * partialDerivativeOfNode
            node.weights[weightIndex] += weightDelta;
          }
        }
      }
      
      sumPresentationError += totalError;
    })
    console.log(k + "/" + presentations, sumPresentationError/trainingData.length);
  }

  console.log(model[model.length - 1].forEach(node => {
    console.log(node.value)
  }))

  niceLogModel(model)

  let correctCount = 0;

  testingData.forEach(exPair => {
    run(model, exPair);
    const expectedResult = [...exPair].slice(4, 7);
    const actualResult = getOutputValues(model)
    console.log(expectedResult, actualResult);

    let topIndex = -1;
    
    actualResult.forEach((number, index) => {
      if (topIndex === -1 || number > actualResult[topIndex]){
        topIndex = index;
      }
    });

    //console.log(topIndex)

    if(expectedResult[topIndex] === 1){
      console.log("Correct");
      correctCount++;
    } else {
      console.log("Incorrect");
    }
  });

  console.log(`Correct/Incorrect ${correctCount}/${testingData.length - correctCount}`)
  console.log(`Accuracy ${(correctCount/testingData.length) * 100}`)

  correctCount = 0;

  validationData.forEach(exPair => {
    run(model, exPair);
    const expectedResult = [...exPair].slice(4, 7);
    const actualResult = getOutputValues(model)
    console.log(expectedResult, actualResult);

    let topIndex = -1;
    
    actualResult.forEach((number, index) => {
      if (topIndex === -1 || number > actualResult[topIndex]){
        topIndex = index;
      }
    });

    //console.log(topIndex)

    if(expectedResult[topIndex] === 1){
      console.log("Correct");
      correctCount++;
    } else {
      console.log("Incorrect");
    }
  });

  console.log(`Correct/Incorrect ${correctCount}/${validationData.length - correctCount}`)
  console.log(`Accuracy ${(correctCount/validationData.length) * 100}`)
}

function getOutputValues(model){
  return model[model.length - 1].reduce((acc, node) => {
    return [...acc, node.value];
  }, []);
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
    //let error = Math.pow(expectedOutput[nodeIndex] - outputLayer[nodeIndex].value, 2);
    let error = expectedOutput[nodeIndex] - outputLayer[nodeIndex].value;


    //console.log(error);
    outputLayer[nodeIndex].error = error;
    //outputLayer[nodeIndex].derivative = slope(sigmoid, error);
    //outputLayer[nodeIndex].derivative = slope(sigmoid, outputLayer[nodeIndex].value);


    // We need to total error of the layer for later
    totalError += outputLayer[nodeIndex].error;
  }

  return totalError;
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

/*
// From: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
//0.21492890298354272
function slope (f, x) {
  return f(x) * (1 - f(x));
}
*/

function niceLogModel(model){
  console.group('model');
  model.forEach((layer, index) => {
    console.group(`layer ${index}`);
    console.log(index, JSON.stringify(layer, null, 2))
    console.group(`layer ${index}`);
  })
  console.groupEnd('model');
}