??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02v2.0.0-rc2-26-g64c3d388??
~
conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:
n
conv2d/biasVarHandleOp*
shape:*
shared_nameconv2d/bias*
dtype0*
_output_shapes
: 
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
shape:* 
shared_nameconv2d_1/kernel*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
r
conv2d_1/biasVarHandleOp*
shape:*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
u
dense/kernelVarHandleOp*
_output_shapes
: *
shape:	? *
shared_namedense/kernel*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	? 
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
: 
x
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*
dtype0*
_output_shapes
: *
shape
: 

q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

: 

p
dense_1/biasVarHandleOp*
shape:
*
shared_namedense_1/bias*
dtype0*
_output_shapes
: 
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

^
totalVarHandleOp*
_output_shapes
: *
shape: *
shared_nametotal*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

NoOpNoOp
?)
ConstConst"/device:CPU:0*?(
value?(B?( B?(
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
	optimizer

signatures
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
~

kernel
bias
_callable_losses
	variables
regularization_losses
trainable_variables
	keras_api
h
_callable_losses
	variables
regularization_losses
 trainable_variables
!	keras_api
~

"kernel
#bias
$_callable_losses
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h
)_callable_losses
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h
._callable_losses
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h
3_callable_losses
4	variables
5regularization_losses
6trainable_variables
7	keras_api
~

8kernel
9bias
:_callable_losses
;	variables
<regularization_losses
=trainable_variables
>	keras_api
h
?_callable_losses
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h
D_callable_losses
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
~

Ikernel
Jbias
K_callable_losses
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
 
 
8
0
1
"2
#3
84
95
I6
J7
 
8
0
1
"2
#3
84
95
I6
J7
?
	variables
Pmetrics
Qnon_trainable_variables
regularization_losses
Rlayer_regularization_losses

Slayers
trainable_variables
 
 
 
?
	variables
Tmetrics
regularization_losses
Unon_trainable_variables
Vlayer_regularization_losses

Wlayers
trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
	variables
Xmetrics
regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
trainable_variables
 
 
 
 
?
	variables
\metrics
regularization_losses
]non_trainable_variables
^layer_regularization_losses

_layers
 trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1
 

"0
#1
?
%	variables
`metrics
&regularization_losses
anon_trainable_variables
blayer_regularization_losses

clayers
'trainable_variables
 
 
 
 
?
*	variables
dmetrics
+regularization_losses
enon_trainable_variables
flayer_regularization_losses

glayers
,trainable_variables
 
 
 
 
?
/	variables
hmetrics
0regularization_losses
inon_trainable_variables
jlayer_regularization_losses

klayers
1trainable_variables
 
 
 
 
?
4	variables
lmetrics
5regularization_losses
mnon_trainable_variables
nlayer_regularization_losses

olayers
6trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
 

80
91
?
;	variables
pmetrics
<regularization_losses
qnon_trainable_variables
rlayer_regularization_losses

slayers
=trainable_variables
 
 
 
 
?
@	variables
tmetrics
Aregularization_losses
unon_trainable_variables
vlayer_regularization_losses

wlayers
Btrainable_variables
 
 
 
 
?
E	variables
xmetrics
Fregularization_losses
ynon_trainable_variables
zlayer_regularization_losses

{layers
Gtrainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1
 

I0
J1
?
L	variables
|metrics
Mregularization_losses
}non_trainable_variables
~layer_regularization_losses

layers
Ntrainable_variables

?0
 
 
F
0
1
2
3
4
5
6
	7

8
9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
 
?
?	variables
?metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?trainable_variables
 

?0
?1
 
 *
dtype0*
_output_shapes
: 
?
serving_default_conv2d_inputPlaceholder*$
shape:?????????*
dtype0*/
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:?????????
*+
_gradient_op_typePartitionedCall-1816*+
f&R$
"__inference_signature_wrapper_1790*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*
_output_shapes
: *+
_gradient_op_typePartitionedCall-1848*&
f!R
__inference__traced_save_1847
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcount*-
config_proto

CPU

GPU2*0J 8*
_output_shapes
: *
Tin
2*+
_gradient_op_typePartitionedCall-1891*)
f$R"
 __inference__traced_restore_1890*
Tout
2??
?

?
?__inference_conv2d_layer_call_and_return_conditional_losses_363

inputs'
#conv2d_readvariableop_conv2d_kernel&
"biasadd_readvariableop_conv2d_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????*
T0*
strides
*
paddingVALIDu
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_conv2d_bias*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
\
@__inference_flatten_layer_call_and_return_conditional_losses_525

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0Z
Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
?????????u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
_output_shapes
:*
T0*
Ne
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
@__inference_dense_1_layer_call_and_return_conditional_losses_801

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
?__inference_dense_1_layer_call_and_return_conditional_losses_51

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

: 
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
@
$__inference_dropout_layer_call_fn_12

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*(
_gradient_op_typePartitionedCall-7*G
fBR@
>__inference_dropout_layer_call_and_return_conditional_losses_6h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
^
?__inference_dropout_layer_call_and_return_conditional_losses_92

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
valueB
 *  ?>*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0_
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*/
_output_shapes
:?????????*
T0*
dtype0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:??????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:?????????*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:?????????*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
'__inference_restored_function_body_1389

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-325*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*A
_output_shapes/
-:+????????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
C
'__inference_restored_function_body_1479

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-313*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_312*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
a
B__inference_dropout_1_layer_call_and_return_conditional_losses_450

inputs
identity?Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:????????? ?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:????????? *
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:????????? R
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*'
_output_shapes
:????????? *
T0a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:????????? *

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?	
\
@__inference_flatten_layer_call_and_return_conditional_losses_423

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskZ
Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
_output_shapes
:*
T0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
&__inference_conv2d_1_layer_call_fn_332

inputs+
'statefulpartitionedcall_conv2d_1_kernel)
%statefulpartitionedcall_conv2d_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*A
_output_shapes/
-:+???????????????????????????**
_gradient_op_typePartitionedCall-325*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
C
'__inference_restored_function_body_1422

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-405*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_404*
Tout
2*-
config_proto

CPU

GPU2*0J 8h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
`
'__inference_restored_function_body_1412

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*)
_gradient_op_typePartitionedCall-93*H
fCRA
?__inference_dropout_layer_call_and_return_conditional_losses_92*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
C
'__inference_restored_function_body_1378

inputs
identity?
PartitionedCallPartitionedCallinputs*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635*
Tout
2*-
config_proto

CPU

GPU2*0J 8*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2**
_gradient_op_typePartitionedCall-636?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
'__inference_restored_function_body_1367

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-364*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_363*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*A
_output_shapes/
-:+????????????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+???????????????????????????*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
H
,__inference_max_pooling2d_1_layer_call_fn_71

inputs
identity?
PartitionedCallPartitionedCallinputs*P
fKRI
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65*
Tout
2*-
config_proto

CPU

GPU2*0J 8*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2*)
_gradient_op_typePartitionedCall-66?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?$
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1717
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1368*0
f+R)
'__inference_restored_function_body_1367?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1379*0
f+R)
'__inference_restored_function_body_1378*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1390*0
f+R)
'__inference_restored_function_body_1389*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*0
f+R)
'__inference_restored_function_body_1400*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1401?
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1423*0
f+R)
'__inference_restored_function_body_1422*
Tout
2?
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*0
f+R)
'__inference_restored_function_body_1435*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*(
_output_shapes
:??????????*+
_gradient_op_typePartitionedCall-1436?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1447*0
f+R)
'__inference_restored_function_body_1446*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2*+
_gradient_op_typePartitionedCall-1458*0
f+R)
'__inference_restored_function_body_1457*
Tout
2?
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1480*0
f+R)
'__inference_restored_function_body_1479*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
*+
_gradient_op_typePartitionedCall-1495*0
f+R)
'__inference_restored_function_body_1494?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
^
@__inference_dropout_layer_call_and_return_conditional_losses_404

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:?????????*
T0c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
? 
?
__inference__traced_save_1847
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_f9ae413612ca45cd970781ab2539b38a/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
?
SaveV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*l
_input_shapes[
Y: :::::	? : : 
:
: : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
?
C
'__inference_dropout_1_layer_call_fn_462

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2**
_gradient_op_typePartitionedCall-457*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_456`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
C
'__inference_restored_function_body_1400

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*J
_output_shapes8
6:4????????????????????????????????????*)
_gradient_op_typePartitionedCall-66*P
fKRI
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
,
__inference_isrlu_109
x
identityA
SignSignx*'
_output_shapes
:????????? *
T0I
IdentityIdentityx*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? :! 

_user_specified_namex
?
_
@__inference_dropout_layer_call_and_return_conditional_losses_392

inputs
identity?Q
dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:??????????
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*/
_output_shapes
:?????????*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:?????????i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:?????????q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?

?
+__inference_sequential_1_layer_call_fn_1775
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*+
_gradient_op_typePartitionedCall-1764*0
f+R)
'__inference_restored_function_body_1763*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2	?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
?
$__inference_dense_1_layer_call_fn_59

inputs*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
*)
_gradient_op_typePartitionedCall-52*H
fCRA
?__inference_dense_1_layer_call_and_return_conditional_losses_51*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
$__inference_conv2d_layer_call_fn_371

inputs)
%statefulpartitionedcall_conv2d_kernel'
#statefulpartitionedcall_conv2d_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_363*
Tout
2*-
config_proto

CPU

GPU2*0J 8*A
_output_shapes/
-:+???????????????????????????*
Tin
2**
_gradient_op_typePartitionedCall-364?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?

?
'__inference_restored_function_body_1763

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*'
_output_shapes
:?????????
*
Tin
2	**
_gradient_op_typePartitionedCall-696*3
f.R,
*__inference_sequential_1_layer_call_fn_695*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
a
E__inference_activation_1_layer_call_and_return_conditional_losses_247

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-110*
fR
__inference_isrlu_109*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? `
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?'
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1693
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1368*0
f+R)
'__inference_restored_function_body_1367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????*
Tin
2*+
_gradient_op_typePartitionedCall-1379*0
f+R)
'__inference_restored_function_body_1378*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1390*0
f+R)
'__inference_restored_function_body_1389*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1401*0
f+R)
'__inference_restored_function_body_1400*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1413*0
f+R)
'__inference_restored_function_body_1412*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2*+
_gradient_op_typePartitionedCall-1436*0
f+R)
'__inference_restored_function_body_1435?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2*+
_gradient_op_typePartitionedCall-1447*0
f+R)
'__inference_restored_function_body_1446?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1458*0
f+R)
'__inference_restored_function_body_1457*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*+
_gradient_op_typePartitionedCall-1470*0
f+R)
'__inference_restored_function_body_1469*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1495*0
f+R)
'__inference_restored_function_body_1494*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_456

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:????????? *
T0[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
^
%__inference_dropout_layer_call_fn_398

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2**
_gradient_op_typePartitionedCall-393*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_392*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
`
'__inference_restored_function_body_1469

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs**
_gradient_op_typePartitionedCall-451*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_450*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?'
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_681

inputs0
,conv2d_statefulpartitionedcall_conv2d_kernel.
*conv2d_statefulpartitionedcall_conv2d_bias4
0conv2d_1_statefulpartitionedcall_conv2d_1_kernel2
.conv2d_1_statefulpartitionedcall_conv2d_1_bias.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-364*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_363*
Tout
2?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-636*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635*
Tout
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2**
_gradient_op_typePartitionedCall-325*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*)
_gradient_op_typePartitionedCall-66*P
fKRI
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65*
Tout
2?
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*G
fBR@
>__inference_dropout_layer_call_and_return_conditional_losses_6*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2*(
_gradient_op_typePartitionedCall-7?
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2**
_gradient_op_typePartitionedCall-424*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_423*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2**
_gradient_op_typePartitionedCall-344*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_343*
Tout
2?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-116*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_115*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_456*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-457?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
*)
_gradient_op_typePartitionedCall-52*H
fCRA
?__inference_dense_1_layer_call_and_return_conditional_losses_51*
Tout
2?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?)
?
__inference__wrapped_model_1668
conv2d_input6
2sequential_1_conv2d_statefulpartitionedcall_args_16
2sequential_1_conv2d_statefulpartitionedcall_args_28
4sequential_1_conv2d_1_statefulpartitionedcall_args_18
4sequential_1_conv2d_1_statefulpartitionedcall_args_25
1sequential_1_dense_statefulpartitionedcall_args_15
1sequential_1_dense_statefulpartitionedcall_args_27
3sequential_1_dense_1_statefulpartitionedcall_args_17
3sequential_1_dense_1_statefulpartitionedcall_args_2
identity??+sequential_1/conv2d/StatefulPartitionedCall?-sequential_1/conv2d_1/StatefulPartitionedCall?*sequential_1/dense/StatefulPartitionedCall?,sequential_1/dense_1/StatefulPartitionedCall?
+sequential_1/conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input2sequential_1_conv2d_statefulpartitionedcall_args_12sequential_1_conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1368*0
f+R)
'__inference_restored_function_body_1367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
*sequential_1/max_pooling2d/PartitionedCallPartitionedCall4sequential_1/conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1379*0
f+R)
'__inference_restored_function_body_1378*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
-sequential_1/conv2d_1/StatefulPartitionedCallStatefulPartitionedCall3sequential_1/max_pooling2d/PartitionedCall:output:04sequential_1_conv2d_1_statefulpartitionedcall_args_14sequential_1_conv2d_1_statefulpartitionedcall_args_2*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1390*0
f+R)
'__inference_restored_function_body_1389*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
,sequential_1/max_pooling2d_1/PartitionedCallPartitionedCall6sequential_1/conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1401*0
f+R)
'__inference_restored_function_body_1400*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
$sequential_1/dropout/PartitionedCallPartitionedCall5sequential_1/max_pooling2d_1/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????*+
_gradient_op_typePartitionedCall-1423*0
f+R)
'__inference_restored_function_body_1422?
$sequential_1/flatten/PartitionedCallPartitionedCall-sequential_1/dropout/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1436*0
f+R)
'__inference_restored_function_body_1435*
Tout
2*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2?
*sequential_1/dense/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/flatten/PartitionedCall:output:01sequential_1_dense_statefulpartitionedcall_args_11sequential_1_dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1447*0
f+R)
'__inference_restored_function_body_1446*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
)sequential_1/activation_1/PartitionedCallPartitionedCall3sequential_1/dense/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1458*0
f+R)
'__inference_restored_function_body_1457*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
&sequential_1/dropout_1/PartitionedCallPartitionedCall2sequential_1/activation_1/PartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2*+
_gradient_op_typePartitionedCall-1480*0
f+R)
'__inference_restored_function_body_1479?
,sequential_1/dense_1/StatefulPartitionedCallStatefulPartitionedCall/sequential_1/dropout_1/PartitionedCall:output:03sequential_1_dense_1_statefulpartitionedcall_args_13sequential_1_dense_1_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
*+
_gradient_op_typePartitionedCall-1495*0
f+R)
'__inference_restored_function_body_1494?
IdentityIdentity5sequential_1/dense_1/StatefulPartitionedCall:output:0,^sequential_1/conv2d/StatefulPartitionedCall.^sequential_1/conv2d_1/StatefulPartitionedCall+^sequential_1/dense/StatefulPartitionedCall-^sequential_1/dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2Z
+sequential_1/conv2d/StatefulPartitionedCall+sequential_1/conv2d/StatefulPartitionedCall2\
,sequential_1/dense_1/StatefulPartitionedCall,sequential_1/dense_1/StatefulPartitionedCall2^
-sequential_1/conv2d_1/StatefulPartitionedCall-sequential_1/conv2d_1/StatefulPartitionedCall2X
*sequential_1/dense/StatefulPartitionedCall*sequential_1/dense/StatefulPartitionedCall: : : : : : : :, (
&
_user_specified_nameconv2d_input: 
?

?
'__inference_restored_function_body_1734

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
_gradient_op_typePartitionedCall-790*3
f.R,
*__inference_sequential_1_layer_call_fn_789*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:?????????
?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
?
'__inference_restored_function_body_1494

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:?????????
*
Tin
2**
_gradient_op_typePartitionedCall-802*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_801*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?
G
+__inference_max_pooling2d_layer_call_fn_738

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-636*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635*
Tout
2*-
config_proto

CPU

GPU2*0J 8*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2?
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
*__inference_sequential_1_layer_call_fn_789

inputs)
%statefulpartitionedcall_conv2d_kernel'
#statefulpartitionedcall_conv2d_bias+
'statefulpartitionedcall_conv2d_1_kernel)
%statefulpartitionedcall_conv2d_1_bias(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*
Tin
2	*'
_output_shapes
:?????????
**
_gradient_op_typePartitionedCall-762*N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_761*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?+
?
 __inference__traced_restore_1890
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias
assignvariableop_8_total
assignvariableop_9_count
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:x
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_5: : : : : : :	 :
 :+ '
%
_user_specified_namefile_prefix: : 
?

?
"__inference_signature_wrapper_1790
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*+
_gradient_op_typePartitionedCall-1779*(
f#R!
__inference__wrapped_model_1668*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2	?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
?
*__inference_sequential_1_layer_call_fn_695
conv2d_input)
%statefulpartitionedcall_conv2d_kernel'
#statefulpartitionedcall_conv2d_bias+
'statefulpartitionedcall_conv2d_1_kernel)
%statefulpartitionedcall_conv2d_1_bias(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_input%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:?????????
**
_gradient_op_typePartitionedCall-682*N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_681*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :, (
&
_user_specified_nameconv2d_input: : : : : : 
?
F
*__inference_activation_1_layer_call_fn_121

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-116*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_115`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
C
'__inference_restored_function_body_1457

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-248*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_247*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
>__inference_dense_layer_call_and_return_conditional_losses_103

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	? i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?
?
#__inference_dense_layer_call_fn_351

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_343*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2**
_gradient_op_typePartitionedCall-344?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
'__inference_restored_function_body_1446

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-104*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_103*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?*
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_761

inputs0
,conv2d_statefulpartitionedcall_conv2d_kernel.
*conv2d_statefulpartitionedcall_conv2d_bias4
0conv2d_1_statefulpartitionedcall_conv2d_1_kernel2
.conv2d_1_statefulpartitionedcall_conv2d_1_bias.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_363*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-364?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-636?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2**
_gradient_op_typePartitionedCall-325*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*)
_gradient_op_typePartitionedCall-66*P
fKRI
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:??????????
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-393*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_392*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*(
_output_shapes
:??????????**
_gradient_op_typePartitionedCall-424*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_423*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias**
_gradient_op_typePartitionedCall-344*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_343*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? ?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2**
_gradient_op_typePartitionedCall-116*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_115?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2*)
_gradient_op_typePartitionedCall-34*J
fERC
A__inference_dropout_1_layer_call_and_return_conditional_losses_33*
Tout
2?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:?????????
*)
_gradient_op_typePartitionedCall-52*H
fCRA
?__inference_dense_1_layer_call_and_return_conditional_losses_51?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?

?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324

inputs)
%conv2d_readvariableop_conv2d_1_kernel(
$biasadd_readvariableop_conv2d_1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*
dtype0*&
_output_shapes
:?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+???????????????????????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_conv2d_1_bias*
dtype0*
_output_shapes
:?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+????????????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
>__inference_dense_layer_call_and_return_conditional_losses_343

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	? i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? t
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
: v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
?
\
>__inference_dropout_layer_call_and_return_conditional_losses_6

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
_
&__inference_dropout_1_layer_call_fn_39

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? *)
_gradient_op_typePartitionedCall-34*J
fERC
A__inference_dropout_1_layer_call_and_return_conditional_losses_33?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
`
A__inference_dropout_1_layer_call_and_return_conditional_losses_33

inputs
identity?Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:????????? ?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*'
_output_shapes
:????????? *
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:????????? R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:????????? a
dropout/mulMulinputsdropout/truediv:z:0*'
_output_shapes
:????????? *
T0o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:????????? i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65

inputs
identity?
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4????????????????????????????????????*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?

?
+__inference_sequential_1_layer_call_fn_1746
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2	*+
_gradient_op_typePartitionedCall-1735*0
f+R)
'__inference_restored_function_body_1734?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_312

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:????????? *
T0[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:????????? *
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
a
E__inference_activation_1_layer_call_and_return_conditional_losses_115

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-110*
fR
__inference_isrlu_109*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:????????? *
Tin
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
A
%__inference_flatten_layer_call_fn_429

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*(
_output_shapes
:??????????**
_gradient_op_typePartitionedCall-424*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_423*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
,
__inference_isrlu_410
x
identityA
SignSignx*
T0*'
_output_shapes
:????????? I
IdentityIdentityx*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :! 

_user_specified_namex
?
C
'__inference_restored_function_body_1435

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*(
_output_shapes
:??????????*
Tin
2**
_gradient_op_typePartitionedCall-526*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_525*
Tout
2a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
M
conv2d_input=
serving_default_conv2d_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?:
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
	optimizer

signatures
	variables
regularization_losses
trainable_variables
	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?7
_tf_keras_sequential?7{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}
?

kernel
bias
_callable_losses
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?
_callable_losses
	variables
regularization_losses
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

"kernel
#bias
$_callable_losses
%	variables
&regularization_losses
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}}
?
)_callable_losses
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
._callable_losses
/	variables
0regularization_losses
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?
3_callable_losses
4	variables
5regularization_losses
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

8kernel
9bias
:_callable_losses
;	variables
<regularization_losses
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
?
?_callable_losses
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
activation"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}}
?
D_callable_losses
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Ikernel
Jbias
K_callable_losses
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
"
	optimizer
-
?serving_default"
signature_map
X
0
1
"2
#3
84
95
I6
J7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
"2
#3
84
95
I6
J7"
trackable_list_wrapper
?
	variables
Pmetrics
Qnon_trainable_variables
regularization_losses
Rlayer_regularization_losses

Slayers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Tmetrics
regularization_losses
Unon_trainable_variables
Vlayer_regularization_losses

Wlayers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
Xmetrics
regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
\metrics
regularization_losses
]non_trainable_variables
^layer_regularization_losses

_layers
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
%	variables
`metrics
&regularization_losses
anon_trainable_variables
blayer_regularization_losses

clayers
'trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*	variables
dmetrics
+regularization_losses
enon_trainable_variables
flayer_regularization_losses

glayers
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/	variables
hmetrics
0regularization_losses
inon_trainable_variables
jlayer_regularization_losses

klayers
1trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4	variables
lmetrics
5regularization_losses
mnon_trainable_variables
nlayer_regularization_losses

olayers
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	? 2dense/kernel
: 2
dense/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
;	variables
pmetrics
<regularization_losses
qnon_trainable_variables
rlayer_regularization_losses

slayers
=trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@	variables
tmetrics
Aregularization_losses
unon_trainable_variables
vlayer_regularization_losses

wlayers
Btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
E	variables
xmetrics
Fregularization_losses
ynon_trainable_variables
zlayer_regularization_losses

{layers
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 
2dense_1/kernel
:
2dense_1/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
L	variables
|metrics
Mregularization_losses
}non_trainable_variables
~layer_regularization_losses

layers
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
	7

8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
__inference__wrapped_model_1668?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+
conv2d_input?????????
?2?
+__inference_sequential_1_layer_call_fn_1775
+__inference_sequential_1_layer_call_fn_1746?
???
FullArgSpec)
args!?
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1717
F__inference_sequential_1_layer_call_and_return_conditional_losses_1693?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
$__inference_conv2d_layer_call_fn_371?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
?__inference_conv2d_layer_call_and_return_conditional_losses_363?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
+__inference_max_pooling2d_layer_call_fn_738?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_conv2d_1_layer_call_fn_332?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
,__inference_max_pooling2d_1_layer_call_fn_71?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
$__inference_dropout_layer_call_fn_12
%__inference_dropout_layer_call_fn_398?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_dropout_layer_call_and_return_conditional_losses_92
@__inference_dropout_layer_call_and_return_conditional_losses_404?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_flatten_layer_call_fn_429?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_flatten_layer_call_and_return_conditional_losses_525?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_dense_layer_call_fn_351?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_dense_layer_call_and_return_conditional_losses_103?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_activation_1_layer_call_fn_121?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_activation_1_layer_call_and_return_conditional_losses_247?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_isrlu_410?
???
FullArgSpec
args?
jx
jalpha
varargs
 
varkw
 
defaults?
`

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dropout_1_layer_call_fn_39
'__inference_dropout_1_layer_call_fn_462?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_1_layer_call_and_return_conditional_losses_312
B__inference_dropout_1_layer_call_and_return_conditional_losses_450?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_dense_1_layer_call_fn_59?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_1_layer_call_and_return_conditional_losses_801?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
6B4
"__inference_signature_wrapper_1790conv2d_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
+__inference_sequential_1_layer_call_fn_1746k"#89IJE?B
;?8
.?+
conv2d_input?????????
p

 
? "??????????
?
__inference__wrapped_model_1668|"#89IJ=?:
3?0
.?+
conv2d_input?????????
? "1?.
,
dense_1!?
dense_1?????????
?
?__inference_conv2d_layer_call_and_return_conditional_losses_363?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1717x"#89IJE?B
;?8
.?+
conv2d_input?????????
p 

 
? "%?"
?
0?????????

? ?
?__inference_dropout_layer_call_and_return_conditional_losses_92l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? w
#__inference_dense_layer_call_fn_351P890?-
&?#
!?
inputs??????????
? "?????????? ?
$__inference_conv2d_layer_call_fn_371?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
G__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
$__inference_dropout_layer_call_fn_12_;?8
1?.
(?%
inputs?????????
p 
? " ??????????y
*__inference_activation_1_layer_call_fn_121K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
+__inference_sequential_1_layer_call_fn_1775k"#89IJE?B
;?8
.?+
conv2d_input?????????
p 

 
? "??????????
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1693x"#89IJE?B
;?8
.?+
conv2d_input?????????
p

 
? "%?"
?
0?????????

? y
&__inference_dropout_1_layer_call_fn_39O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
@__inference_flatten_layer_call_and_return_conditional_losses_525a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
@__inference_dropout_layer_call_and_return_conditional_losses_404l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
%__inference_dropout_layer_call_fn_398_;?8
1?.
(?%
inputs?????????
p
? " ??????????c
__inference_isrlu_410J.?+
$?!
?
x????????? 
`
? "?????????? ?
,__inference_max_pooling2d_1_layer_call_fn_71?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_dropout_1_layer_call_and_return_conditional_losses_450\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
&__inference_conv2d_1_layer_call_fn_332?"#I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????z
'__inference_dropout_1_layer_call_fn_462O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
"__inference_signature_wrapper_1790?"#89IJM?J
? 
C?@
>
conv2d_input.?+
conv2d_input?????????"1?.
,
dense_1!?
dense_1?????????
?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_324?"#I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? }
%__inference_flatten_layer_call_fn_429T7?4
-?*
(?%
inputs?????????
? "????????????
@__inference_dense_1_layer_call_and_return_conditional_losses_801\IJ/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????

? ?
+__inference_max_pooling2d_layer_call_fn_738?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_635?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_dropout_1_layer_call_and_return_conditional_losses_312\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
E__inference_activation_1_layer_call_and_return_conditional_losses_247X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? w
$__inference_dense_1_layer_call_fn_59OIJ/?,
%?"
 ?
inputs????????? 
? "??????????
?
>__inference_dense_layer_call_and_return_conditional_losses_103]890?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? 