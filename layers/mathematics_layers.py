import logging
import numpy as np
import tensorflow as tf

from utils.op_registry import OPERATOR
from layers import dimension_utils
import keras

LOG = logging.getLogger("calculations_layers :")

def np2tf(x):
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x, False
    return x, True

def match_tensor(x1:tf.Tensor or np.ndarray, x2:tf.Tensor or np.ndarray):
    
    x1, f1 = np2tf(x1)
    x2, f2 = np2tf(x2)

    # no need to transpose if all var are tensor, we assume tensor are computed by gragh.
    if f1 and f2:
        return x1, x2
    
    # ensure tensor is set to x1, weights set to x2
    if f2:
        x1, x2 = x2, x1

    # if x1.shape.ndims != x2.shape.ndims:
    #     while x2.shape.ndims < x1.shape.ndims:
    #         x2 = tf.expand_dims(x2, axis=0)
    if len(x1.shape) != len(x2.shape):
        while len(x2.shape) < len(x1.shape):
            x2 = tf.expand_dims(x2, axis=0)
    
    new_shape = dimension_utils.shape_NCD_to_NDC_format([i for i in range(len(x2.shape))])
    x2 = tf.transpose(x2, new_shape)
    return (x2, x1) if f2 else (x1, x2)


@OPERATOR.register_operator("Add")
class TFAdd(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.add(args[0], args[1])
        return self.first_operand + self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("Sub")
class TFSub(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.subtract(args[0], args[1])
        return self.first_operand - self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("Mul")
class TFMul(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.multiply(args[0], args[1])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("Div")
class TFDiv(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.divide(args[0], args[1])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config
    
@OPERATOR.register_operator("Equal")
class TFEqual(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)

    def call(self, *args, **kwargs):
        return keras.ops.equal(args[0], args[1])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config
    
@OPERATOR.register_operator("Not")
class TFNot(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()


    def call(self,input, *args, **kwargs):
        return keras.ops.logical_not(input)

@OPERATOR.register_operator("Less")
class TFLess(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.less(args[0], args[1])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config
    
@OPERATOR.register_operator("Greater")
class TFGreater(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.first_operand = tensor_grap[node_inputs[0]] if node_inputs[0] in tensor_grap else node_weights[node_inputs[0]]
        self.second_operand = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.first_operand, self.second_operand = match_tensor(self.first_operand, self.second_operand)


    def call(self, *args, **kwargs):
        return keras.ops.greater(args[0], args[1])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand,
            'node_attribute':self.node_attribute
        })
        return config

    
@OPERATOR.register_operator("Where")
class TFWhere(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.true_value = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]
        self.false_value = tensor_grap[node_inputs[2]] if node_inputs[2] in tensor_grap else node_weights[node_inputs[2]]
        self.true_value, self.false_value = match_tensor(self.true_value, self.false_value)


    def call(self, *args, **kwargs):
        return keras.ops.where(args[0], args[1], args[2])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "true_value": self.true_value,
             "false_value": self.false_value,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("ConstantOfShape")
class TFConstantOfShape(keras.layers.Layer):
    def __init__(self,tensor_grap,  node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
    def call(self, inputs,*args, **kwargs):
        return keras.ops.full(inputs, self.node_attribute['value'][0])
        return self.first_operand * self.second_operand
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("Abs")
class TFAbs(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,input,  *args, **kwargs):
        return keras.ops.absolute(input)

@OPERATOR.register_operator("And")
class TFAnd(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,  *args, **kwargs):
        return keras.ops.logical_and(args[0], args[1])

@OPERATOR.register_operator("Or")
class TFOr(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self,  *args, **kwargs):
        return keras.ops.logical_or(args[0], args[1])

@OPERATOR.register_operator("MatMul")
class TFMatMul(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs

        if node_inputs[0] in tensor_grap:
            self.first_operand = tensor_grap[node_inputs[0]]
            print('Matmul first: ', self.first_operand)
            new_shape = [0, len(self.first_operand.shape)-1] + [i for i in range(1, len(self.first_operand.shape)-1)]
            self.first_operand = keras.ops.transpose(self.first_operand, new_shape)
        else:
            self.first_operand = node_weights[node_inputs[0]]

        if node_inputs[1] in tensor_grap:
            self.second_operand = tensor_grap[node_inputs[1]]
            new_shape = [0, len(self.second_operand.shape)-1] + [i for i in range(1, len(self.second_operand.shape)-1)]
            self.second_operand = keras.ops.transpose(self.second_operand, new_shape)
        else:
            self.second_operand = node_weights[node_inputs[1]]

    def call(self, *args, **kwargs):
        out = keras.ops.matmul(self.first_operand, self.second_operand)
        out = dimension_utils.tensor_NCD_to_NDC_format(out)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            "first_operand": self.first_operand,
             "second_operand": self.second_operand
        })
        return config

@OPERATOR.register_operator("Pow")
class TFPow(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.power_index = node_weights[node_inputs[1]]

    def call(self, inputs, *args, **kwargs):
        return keras.ops.power(inputs, self.power_index)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs
        })
        return config

@OPERATOR.register_operator("Reciprocal")
class TFReciprocal(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return keras.ops.reciprocal(inputs)

@OPERATOR.register_operator("Sqrt")
class TFSqrt(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs, *args, **kwargs):
        return keras.ops.sqrt(inputs)

@OPERATOR.register_operator("Exp")
class TFExp(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def cal(self, inputs, *args, **kwargs):
        return keras.ops.exp(inputs)

@OPERATOR.register_operator("Log")
class TFLog(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.log(inputs)

@OPERATOR.register_operator("Floor")
class TFFloor(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.floor(inputs)
    
@OPERATOR.register_operator("Ceil")
class TFCeil(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return keras.ops.ceil(inputs)
    
@OPERATOR.register_operator("Shape")
class TFShape(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        print('value shape: ', keras.ops.shape(inputs))
        print('type shape: ', type(keras.ops.shape(inputs)))
        print('tensor real final: ',keras.ops.shape(inputs)[1])
        print(keras.ops.convert_to_tensor(np.array([1,3,1])))
        print("must 3: ", keras.ops.array([*keras.ops.shape(inputs)])[1])
        print('compare: ', [*keras.ops.shape(inputs)]==[1,3,1])
        # return keras.ops.array([1,3,1])
        return keras.ops.array([*keras.ops.shape(inputs)])
        return keras.ops.shape(inputs)
    

@OPERATOR.register_operator("ReduceSum")
class TFReduceSum(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute
        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def call(self, inputs, *args, **kwargs):
        return keras.ops.sum(inputs, axis = self.axes, keepdims=self.keep_dims)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config

# @OPERATOR.register_operator("ReduceMean")
# class TFReduceMean():
#     def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
#         super().__init__()
#         self.keep_dims = node_attribute.get("keepdims", 1) == 1
#         input_shape_len = len(tensor_grap[node_inputs[0]].shape)
#         self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

#     def __call__(self, inputs, *args, **kwargs):
#         return tf.math.reduce_mean(inputs, axis=self.axes, keepdims=self.keep_dims)

@OPERATOR.register_operator("ReduceMax")
class TFReduceMax(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def call(self, inputs, *args, **kwargs):
        return keras.ops.max(inputs, axis=self.axes, keepdims=self.keep_dims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config

@OPERATOR.register_operator("ReduceMin")
class TFReduceMin(keras.layers.Layer):
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.tensor_grap = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.node_attribute = node_attribute

        self.keep_dims = node_attribute.get("keepdims", 1) == 1
        input_shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.axes = [dimension_utils.channel_to_last_dimension(i) if i >=0 else dimension_utils.channel_to_last_dimension(input_shape_len + i) for i in node_attribute.get("axes", [-1])]

    def call(self, inputs, *args, **kwargs):
        return keras.ops.min(inputs, axis=self.axes, keepdims=self.keep_dims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tensor_grap":self.tensor_grap,
            'node_weights':self.node_weights,
            'node_inputs':self.node_inputs,
            'node_attribute':self.node_attribute
        })
        return config
    



@OPERATOR.register_operator("ArgMax")
class TFArgMax():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("ArgMin")
class TFArgMin():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.keepdims = node_attribute.get("keepdims", 1) == 1

    def __call__(self, inputs, *args, **kwargs):
        _inputs = tf.argmax(inputs, axis=self.axis)
        if self.keepdims:
            _inputs = tf.expand_dims(_inputs, axis=self.axis)
        return _inputs

@OPERATOR.register_operator("Erf")
class TFErf():
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, inputs):
        inputs = tf.math.erf(inputs)
        return inputs