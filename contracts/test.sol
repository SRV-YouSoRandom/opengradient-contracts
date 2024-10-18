// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "opengradient-neuroml/src/OGInference.sol";

contract Test {

    string public resultString;
    TensorLib.Number public resultNumber;
    
    // Constants to avoid magic strings/numbers
    string constant private MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct";
    string constant private LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct";
    string constant private STOP_SEQUENCE = "<end>";
    string constant private INPUT_TENSOR_NAME = "input";
    uint256 constant private MAX_TOKENS = 1000;
    
    // Common function to create model input
    function createModelInput() private pure returns (ModelInput memory) {
        ModelInput memory modelInput = ModelInput(
            new TensorLib.MultiDimensionalNumberTensor[](1),
            new TensorLib.StringTensor[](0)
        );
        
        TensorLib.Number[] memory numbers = new TensorLib.Number[](2);
        numbers[0] = TensorLib.Number(7286679744720459, 17); // 0.07286679744720459
        numbers[1] = TensorLib.Number(4486280083656311, 16); // 0.4486280083656311
        modelInput.numbers[0] = TensorLib.numberTensor1D(INPUT_TENSOR_NAME, numbers);
        
        return modelInput;
    }
    
    // Common function to process model output
    function processModelOutput(ModelOutput memory output) private {
        resultNumber = output.is_simulation_result ? TensorLib.Number(0, 0) : output.numbers[0].values[0];
    }
    
    function run() public {
        ModelOutput memory output = OG_INFERENCE_CONTRACT.runModelInference(
            ModelInferenceRequest(ModelInferenceMode.ZK, MODEL_ID, createModelInput())
        );
        processModelOutput(output);
    }
    
    function runVanilla() public {
        ModelOutput memory output = OG_INFERENCE_CONTRACT.runModelInference(
            ModelInferenceRequest(
                ModelInferenceMode.VANILLA,
                MODEL_ID,
                createModelInput()
            )
        );
        processModelOutput(output);
    }
    
    // Common function to create LLM stop sequence array
    function createStopSequence() private pure returns (string[] memory) {
        string[] memory stopSequence = new string[](1);
        stopSequence[0] = STOP_SEQUENCE;
        return stopSequence;
    }
    
    function runLlm() public {
        LlmResponse memory llmResult = OG_INFERENCE_CONTRACT.runLLMInference(
            LlmInferenceRequest(
                LlmInferenceMode.VANILLA,
                LLM_MODEL,
                "I see trees of green, red roses too, I see them bloom for me and you\n<start>",
                uint32(MAX_TOKENS),
                createStopSequence(),
                0
            )
        );
        resultString = llmResult.answer;
    }
    
    function runTee() public {
        LlmResponse memory llmResult = OG_INFERENCE_CONTRACT.runLLMInference(
            LlmInferenceRequest(
                LlmInferenceMode.TEE,
                LLM_MODEL,
                "And I think to myself, What a wonderful world\n<start>",
                uint32(MAX_TOKENS),
                createStopSequence(),
                0
            )
        );
        resultString = llmResult.answer;
    }
    
    function result() public view returns (int128, int128) {
        return (resultNumber.value, resultNumber.decimals);
    }
    
    // Add error handling
    error ModelInferenceError();
    error LlmInferenceError();
    
    // Add events for monitoring
    event ModelInferenceCompleted(bool isSimulation, int128 value, int128 decimals);
    event LlmInferenceCompleted(string result);
}