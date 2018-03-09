const R = require('ramda')

// "Toyota Camry 2014" -> "______ Camry 2014"
function replaceKeyWordsWithSymbol(text, keywords, symbol = '_') {
  console.assert(symbol && typeof symbol === 'string' && symbol.length === 1, "The symbol has to be one character")
  return keywords.reduce((result, keyword) => {
    return result.replace(new RegExp(keyword, 'g'), new Array(keyword.length+1).join(symbol) )
  }, text);
}


// 26 characters, 10 digits, -, and other characters including space are considered as undefined,
function createSlugDictionary() {
  const dictionary = {};
  const aCode = 'a'.charCodeAt(0)
  for(var i=0;i<26;i++) {
    dictionary[String.fromCharCode(aCode + i)] = i;
  }
  for(var i=0;i<10;i++) {
    dictionary[i+''] = 26 + i;
  }
  dictionary['-'] = 36;

  return dictionary;
}

// given a character, map it into a vector in the dictionary space.
function charToVector(ch, dictionary) {
  return intToVector(dictionary[ch] || 0, Object.keys(dictionary).length)
}

// when dimention = 2: 0 => [0,0], 1 => [1,0], 2 => [0, 1]
// when dimention = 3: 0 => [0,0,0], 1 => [1,0,0], 2 => [0, 1, 0], 3 => [0, 0, 1]
function intToVector(value, dimension) {
    console.assert(value >= 0 && value <= dimension, "valid values are between: 0 and ", dimension, " inclusively")
    var index = value - 1;
    return vector = [...new Array(dimension)].map((x,i)=> i == index ? 1 : 0);
}


function distance(v1, v2) {
  return R.compose(R.sum, R.map(pair =>
    Math.pow(pair[0]-pair[1],2)
  )) (R.zip(v1, v2));
}

function displayResultsInCap(contents, results, threshold = 0.5) {
  var output = "";
  var prev = 0;
  var upperCase = false;
  R.forEach(pair => {
  	var ch = pair[0];
  	const b = pair[1] > threshold;
  	if(b) {
  		ch = ch.toUpperCase();
  	}
  	output += ch;
  })(R.zip(contents, results))

  return output;
}

module.exports = {
  replaceKeyWordsWithSymbol,
  createSlugDictionary,
  charToVector,
  intToVector,
  distance,
  displayResultsInCap
}
