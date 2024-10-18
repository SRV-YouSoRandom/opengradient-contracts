const fs = require('fs');
const path = require('path');

// Read the compiled artifact
const artifactPath = path.join(__dirname, 'out', 'OGInference.sol', 'OGInference.json');
const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));

module.exports = {
  abi: artifact.abi,
  bytecode: artifact.bytecode,
};