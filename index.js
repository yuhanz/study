const synaptic = require('synaptic');
const R = require('ramda')
var fs = require('fs');
const util = require('./util.js');


// var contents = fs.readFileSync('./data/makes.txt', 'utf8');
// var makes = contents.toLowerCase().split(',');
makes = ["toyota"];

contents = fs.readFileSync('./data/make-models.txt', 'utf8');
contents = contents.toLowerCase();

const dictionary = util.createSlugDictionary();
console.log("dictionary: ", dictionary)

const OUTPUT_DIMENSION = 1;
const INPUT_DIMENSION = Object.keys(dictionary).length;
const HIDDEN_SIZE = 20;

var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

// 26 characters + 10 digits +
var LSTM = new Architect.LSTM(INPUT_DIMENSION, HIDDEN_SIZE, OUTPUT_DIMENSION);

var contentsTruth = R.map(c => c == '_' ? 1 : 0)(util.replaceKeyWordsWithSymbol(contents, makes, '_'))

// console.log(util.displayResultsInCap(contents, contentsTruth))

console.log("Training...")

R.times(R.identity, 5).forEach(()=> {

	var s = R.compose(R.sum, R.map(pair => {
		const char = pair[0];
		const expectedOutput = pair[1];

		const input = util.charToVector(char, dictionary);
		const expectedOutputVector = util.intToVector(expectedOutput, OUTPUT_DIMENSION)

		var output = LSTM.activate(input)
		LSTM.propagate(0.01, expectedOutputVector)

		const dist = util.distance(output, expectedOutputVector);
		// console.log("output: ", output, " expected: ", expectedOutputVector);
		// console.log("distance: ", dist);
		return dist

	}))(R.zip(contents, contentsTruth))
	console.log("dist: ", s)
})


console.log("Classifying...")


function classifyText(contents) {
	contents = contents.toLowerCase();
	return R.map(ch => {
		const input = util.charToVector(ch, dictionary);
		const r = LSTM.activate(input);
		// console.log(r);
		return r;
	})(contents)
}

// contents = fs.readFileSync('./data/make-model-trims.txt', 'utf8');
// contents = contents.toLowerCase();
//
// contents = contents.substring(0,1000)

contents = "do you believe that toyota is totally affordable.";
var results = classifyText(contents)
console.log("results: ", results);
console.log( output = util.displayResultsInCap(contents, results, 0.08));



// fs.writeFile("/tmp/result.txt", output);
