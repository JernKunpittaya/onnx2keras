This branch differs a lot from original already, so instead of explaining how it's different, let's just explain how to use this

-Edit mathematics_layers, make each math function become their own layer because 'keras2circom' will only look at model.layers, so need to make sure our math computation is included in 'layer'
-I just edit a few functions, particularly look at TFReducesum, and TFLog since it is used in example
-The flow start from generating onnx file in onnx_sum(log) file
-Then in command line, run
`python converter.py --weights "./example/sum(log)/sum(log).onnx" --outpath "./example/sum(log)" --formats "keras"`
-Then in read_keras_sum(log) notebook, it downloads this keras format and run in over same input, seeing that it get the same result as onnx.

Note: I still didnt modify some parts of math layer, particularly abt Adding, Subtracting multiple inputs stuffs, will update later
