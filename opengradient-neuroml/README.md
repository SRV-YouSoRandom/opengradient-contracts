# OpenGradient NeuroML 

Read full documentation [here](https://docs.opengradient.ai/developers/neuro_ml/).

## Example

```solidity
pragma solidity ^0.8.10;

import "opengradient-neuroml/src/OGInference.sol";

contract Test {
    ModelInput memory modelInput = ModelInput(
        new TensorLib.MultiDimensionalNumberTensor[](1),
        new TensorLib.StringTensor[](0));

    TensorLib.Number[] memory numbers = new TensorLib.Number[](2);
    numbers[0] = TensorLib.Number(7286679744720459, 17); // 0.07286679744720459
    numbers[1] = TensorLib.Number(4486280083656311, 16); // 0.4486280083656311
    modelInput.numbers[0] = TensorLib.numberTensor1D("input", numbers);

    ModelOutput memory output = OG_INFERENCE_CONTRACT.runModelInference(
        ModelInferenceRequest(
            ModelInferenceMode.ZK, "QmbbzDwqSxZSgkz1EbsNHp2mb67rYeUYHYWJ4wECE24S7A", 
            modelInput));
}
```
