ٜ

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
shapeshape?"serve*1.15.02v1.15.0-0-g590d6eef7e8??
~
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel*
dtype0*
_output_shapes
: *
shape:
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:
n
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:
?
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
r
conv2d_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	? *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	? 
l

dense/biasVarHandleOp*
shared_name
dense/bias*
dtype0*
_output_shapes
: *
shape: 
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
dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

x
training/Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
|
training/Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
|
training/Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
z
training/Adam/decayVarHandleOp*
shape: *$
shared_nametraining/Adam/decay*
dtype0*
_output_shapes
: 
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
?
training/Adam/learning_rateVarHandleOp*
dtype0*
_output_shapes
: *
shape: *,
shared_nametraining/Adam/learning_rate
?
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
training/Adam/conv2d/kernel/mVarHandleOp*
shape:*.
shared_nametraining/Adam/conv2d/kernel/m*
dtype0*
_output_shapes
: 
?
1training/Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/m*
dtype0*&
_output_shapes
:
?
training/Adam/conv2d/bias/mVarHandleOp*,
shared_nametraining/Adam/conv2d/bias/m*
dtype0*
_output_shapes
: *
shape:
?
/training/Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/m*
dtype0*
_output_shapes
:
?
training/Adam/conv2d_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*0
shared_name!training/Adam/conv2d_1/kernel/m
?
3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/m*
dtype0*&
_output_shapes
:
?
training/Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_nametraining/Adam/conv2d_1/bias/m
?
1training/Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/m*
dtype0*
_output_shapes
:
?
training/Adam/dense/kernel/mVarHandleOp*
shape:	? *-
shared_nametraining/Adam/dense/kernel/m*
dtype0*
_output_shapes
: 
?
0training/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/m*
dtype0*
_output_shapes
:	? 
?
training/Adam/dense/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *+
shared_nametraining/Adam/dense/bias/m
?
.training/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/m*
dtype0*
_output_shapes
: 
?
training/Adam/dense_1/kernel/mVarHandleOp*/
shared_name training/Adam/dense_1/kernel/m*
dtype0*
_output_shapes
: *
shape
: 

?
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
dtype0*
_output_shapes

: 

?
training/Adam/dense_1/bias/mVarHandleOp*
shape:
*-
shared_nametraining/Adam/dense_1/bias/m*
dtype0*
_output_shapes
: 
?
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
dtype0*
_output_shapes
:

?
training/Adam/conv2d/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*.
shared_nametraining/Adam/conv2d/kernel/v
?
1training/Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/kernel/v*
dtype0*&
_output_shapes
:
?
training/Adam/conv2d/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*,
shared_nametraining/Adam/conv2d/bias/v
?
/training/Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d/bias/v*
dtype0*
_output_shapes
:
?
training/Adam/conv2d_1/kernel/vVarHandleOp*
shape:*0
shared_name!training/Adam/conv2d_1/kernel/v*
dtype0*
_output_shapes
: 
?
3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
:
?
training/Adam/conv2d_1/bias/vVarHandleOp*
shape:*.
shared_nametraining/Adam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 
?
1training/Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2d_1/bias/v*
dtype0*
_output_shapes
:
?
training/Adam/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	? *-
shared_nametraining/Adam/dense/kernel/v
?
0training/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/kernel/v*
dtype0*
_output_shapes
:	? 
?
training/Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *+
shared_nametraining/Adam/dense/bias/v
?
.training/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense/bias/v*
dtype0*
_output_shapes
: 
?
training/Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
: 
*/
shared_name training/Adam/dense_1/kernel/v
?
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*
dtype0*
_output_shapes

: 

?
training/Adam/dense_1/bias/vVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/v*
dtype0*
_output_shapes
: *
shape:

?
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
??
ConstConst"/device:CPU:0*??
value?>B?> B?>
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
~

kernel
bias
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
h
_callable_losses
regularization_losses
trainable_variables
 	variables
!	keras_api
~

"kernel
#bias
$_callable_losses
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h
)_callable_losses
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h
._callable_losses
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h
3_callable_losses
4regularization_losses
5trainable_variables
6	variables
7	keras_api
~

8kernel
9bias
:_callable_losses
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h
?_callable_losses
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h
D_callable_losses
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
~

Ikernel
Jbias
K_callable_losses
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem?m?"m?#m?8m?9m?Im?Jm?v?v?"v?#v?8v?9v?Iv?Jv?
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
regularization_losses
Umetrics
trainable_variables
	variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
 
 
 
 
?
regularization_losses
Ymetrics
trainable_variables
	variables
Zlayer_regularization_losses

[layers
\non_trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
regularization_losses
]metrics
trainable_variables
	variables
^layer_regularization_losses

_layers
`non_trainable_variables
 
 
 
 
?
regularization_losses
ametrics
trainable_variables
 	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

"0
#1

"0
#1
?
%regularization_losses
emetrics
&trainable_variables
'	variables
flayer_regularization_losses

glayers
hnon_trainable_variables
 
 
 
 
?
*regularization_losses
imetrics
+trainable_variables
,	variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
 
 
 
 
?
/regularization_losses
mmetrics
0trainable_variables
1	variables
nlayer_regularization_losses

olayers
pnon_trainable_variables
 
 
 
 
?
4regularization_losses
qmetrics
5trainable_variables
6	variables
rlayer_regularization_losses

slayers
tnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

80
91

80
91
?
;regularization_losses
umetrics
<trainable_variables
=	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
 
 
 
 
?
@regularization_losses
ymetrics
Atrainable_variables
B	variables
zlayer_regularization_losses

{layers
|non_trainable_variables
 
 
 
 
?
Eregularization_losses
}metrics
Ftrainable_variables
G	variables
~layer_regularization_losses

layers
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

I0
J1

I0
J1
?
Lregularization_losses
?metrics
Mtrainable_variables
N	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
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
 
?

?total

?count
?
_fn_kwargs
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

?0
?1
?
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
 
 
 

?0
?1
??
VARIABLE_VALUEtraining/Adam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
_gradient_op_typePartitionedCall-1418*+
f&R$
"__inference_signature_wrapper_1068*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1training/Adam/conv2d/kernel/m/Read/ReadVariableOp/training/Adam/conv2d/bias/m/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/m/Read/ReadVariableOp1training/Adam/conv2d_1/bias/m/Read/ReadVariableOp0training/Adam/dense/kernel/m/Read/ReadVariableOp.training/Adam/dense/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp1training/Adam/conv2d/kernel/v/Read/ReadVariableOp/training/Adam/conv2d/bias/v/Read/ReadVariableOp3training/Adam/conv2d_1/kernel/v/Read/ReadVariableOp1training/Adam/conv2d_1/bias/v/Read/ReadVariableOp0training/Adam/dense/kernel/v/Read/ReadVariableOp.training/Adam/dense/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-1471*&
f!R
__inference__traced_save_1470*
Tout
2*-
config_proto

CPU

GPU2*0J 8*,
Tin%
#2!	*
_output_shapes
: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcounttraining/Adam/conv2d/kernel/mtraining/Adam/conv2d/bias/mtraining/Adam/conv2d_1/kernel/mtraining/Adam/conv2d_1/bias/mtraining/Adam/dense/kernel/mtraining/Adam/dense/bias/mtraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/bias/mtraining/Adam/conv2d/kernel/vtraining/Adam/conv2d/bias/vtraining/Adam/conv2d_1/kernel/vtraining/Adam/conv2d_1/bias/vtraining/Adam/dense/kernel/vtraining/Adam/dense/bias/vtraining/Adam/dense_1/kernel/vtraining/Adam/dense_1/bias/v*-
config_proto

CPU

GPU2*0J 8*
_output_shapes
: *+
Tin$
"2 *+
_gradient_op_typePartitionedCall-1577*)
f$R"
 __inference__traced_restore_1576*
Tout
2??
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_1345

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
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682

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
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?*
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1002

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
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias**
_gradient_op_typePartitionedCall-668*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_661*
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
_gradient_op_typePartitionedCall-689*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682*
Tout
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias**
_gradient_op_typePartitionedCall-714*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707*
Tout
2*-
config_proto

CPU

GPU2*0J 8*/
_output_shapes
:?????????*
Tin
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-735*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
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
:?????????**
_gradient_op_typePartitionedCall-785*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_773?
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-818*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_811*
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
:???????????
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias**
_gradient_op_typePartitionedCall-843*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_836*
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
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-867*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_860*
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
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall**
_gradient_op_typePartitionedCall-907*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_895*
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
:????????? ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*
Tin
2*'
_output_shapes
:?????????
**
_gradient_op_typePartitionedCall-939*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_932*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?z
?
 __inference__traced_restore_1576
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias)
%assignvariableop_8_training_adam_iter+
'assignvariableop_9_training_adam_beta_1,
(assignvariableop_10_training_adam_beta_2+
'assignvariableop_11_training_adam_decay3
/assignvariableop_12_training_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count5
1assignvariableop_15_training_adam_conv2d_kernel_m3
/assignvariableop_16_training_adam_conv2d_bias_m7
3assignvariableop_17_training_adam_conv2d_1_kernel_m5
1assignvariableop_18_training_adam_conv2d_1_bias_m4
0assignvariableop_19_training_adam_dense_kernel_m2
.assignvariableop_20_training_adam_dense_bias_m6
2assignvariableop_21_training_adam_dense_1_kernel_m4
0assignvariableop_22_training_adam_dense_1_bias_m5
1assignvariableop_23_training_adam_conv2d_kernel_v3
/assignvariableop_24_training_adam_conv2d_bias_v7
3assignvariableop_25_training_adam_conv2d_1_kernel_v5
1assignvariableop_26_training_adam_conv2d_1_bias_v4
0assignvariableop_27_training_adam_dense_kernel_v2
.assignvariableop_28_training_adam_dense_bias_v6
2assignvariableop_29_training_adam_dense_1_kernel_v4
0assignvariableop_30_training_adam_dense_1_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*-
dtypes#
!2	*?
_output_shapes~
|:::::::::::::::::::::::::::::::L
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

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
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
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_training_adam_iterIdentity_8:output:0*
dtype0	*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp'assignvariableop_9_training_adam_beta_1Identity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp(assignvariableop_10_training_adam_beta_2Identity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_training_adam_decayIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_training_adam_learning_rateIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:{
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_training_adam_conv2d_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_training_adam_conv2d_bias_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp3assignvariableop_17_training_adam_conv2d_1_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp1assignvariableop_18_training_adam_conv2d_1_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp0assignvariableop_19_training_adam_dense_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_training_adam_dense_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_training_adam_dense_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_training_adam_dense_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_training_adam_conv2d_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_training_adam_conv2d_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp3assignvariableop_25_training_adam_conv2d_1_kernel_vIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0?
AssignVariableOp_26AssignVariableOp1assignvariableop_26_training_adam_conv2d_1_bias_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_training_adam_dense_kernel_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_training_adam_dense_bias_vIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_training_adam_dense_1_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_training_adam_dense_1_bias_vIdentity_30:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2: : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
?	
]
A__inference_flatten_layer_call_and_return_conditional_losses_1262

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
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
T0*
N*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
G
+__inference_max_pooling2d_layer_call_fn_692

inputs
identity?
PartitionedCallPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*J
_output_shapes8
6:4????????????????????????????????????*
Tin
2**
_gradient_op_typePartitionedCall-689*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682*
Tout
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
G
+__inference_activation_1_layer_call_fn_1294

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-867*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_860*
Tout
2*-
config_proto

CPU

GPU2*0J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
+__inference_sequential_1_layer_call_fn_1215

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
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*+
_gradient_op_typePartitionedCall-1042*O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_1041*
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
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :& "
 
_user_specified_nameinputs: 
?B
?
__inference__traced_save_1470
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_training_adam_conv2d_kernel_m_read_readvariableop:
6savev2_training_adam_conv2d_bias_m_read_readvariableop>
:savev2_training_adam_conv2d_1_kernel_m_read_readvariableop<
8savev2_training_adam_conv2d_1_bias_m_read_readvariableop;
7savev2_training_adam_dense_kernel_m_read_readvariableop9
5savev2_training_adam_dense_bias_m_read_readvariableop=
9savev2_training_adam_dense_1_kernel_m_read_readvariableop;
7savev2_training_adam_dense_1_bias_m_read_readvariableop<
8savev2_training_adam_conv2d_kernel_v_read_readvariableop:
6savev2_training_adam_conv2d_bias_v_read_readvariableop>
:savev2_training_adam_conv2d_1_kernel_v_read_readvariableop<
8savev2_training_adam_conv2d_1_bias_v_read_readvariableop;
7savev2_training_adam_dense_kernel_v_read_readvariableop9
5savev2_training_adam_dense_bias_v_read_readvariableop=
9savev2_training_adam_dense_1_kernel_v_read_readvariableop;
7savev2_training_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_217b615c76cd47e2bfa00c80ab1981af/part*
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_training_adam_conv2d_kernel_m_read_readvariableop6savev2_training_adam_conv2d_bias_m_read_readvariableop:savev2_training_adam_conv2d_1_kernel_m_read_readvariableop8savev2_training_adam_conv2d_1_bias_m_read_readvariableop7savev2_training_adam_dense_kernel_m_read_readvariableop5savev2_training_adam_dense_bias_m_read_readvariableop9savev2_training_adam_dense_1_kernel_m_read_readvariableop7savev2_training_adam_dense_1_bias_m_read_readvariableop8savev2_training_adam_conv2d_kernel_v_read_readvariableop6savev2_training_adam_conv2d_bias_v_read_readvariableop:savev2_training_adam_conv2d_1_kernel_v_read_readvariableop8savev2_training_adam_conv2d_1_bias_v_read_readvariableop7savev2_training_adam_dense_kernel_v_read_readvariableop5savev2_training_adam_dense_bias_v_read_readvariableop9savev2_training_adam_dense_1_kernel_v_read_readvariableop7savev2_training_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
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
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::	? : : 
:
: : : : : : : :::::	? : : 
:
:::::	? : : 
:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : :  :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : 
?=
?
__inference__wrapped_model_646
conv2d_input;
7sequential_1_conv2d_conv2d_readvariableop_conv2d_kernel:
6sequential_1_conv2d_biasadd_readvariableop_conv2d_bias?
;sequential_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel>
:sequential_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias9
5sequential_1_dense_matmul_readvariableop_dense_kernel8
4sequential_1_dense_biasadd_readvariableop_dense_bias=
9sequential_1_dense_1_matmul_readvariableop_dense_1_kernel<
8sequential_1_dense_1_biasadd_readvariableop_dense_1_bias
identity??*sequential_1/conv2d/BiasAdd/ReadVariableOp?)sequential_1/conv2d/Conv2D/ReadVariableOp?,sequential_1/conv2d_1/BiasAdd/ReadVariableOp?+sequential_1/conv2d_1/Conv2D/ReadVariableOp?)sequential_1/dense/BiasAdd/ReadVariableOp?(sequential_1/dense/MatMul/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7sequential_1_conv2d_conv2d_readvariableop_conv2d_kernel*
dtype0*&
_output_shapes
:?
sequential_1/conv2d/Conv2DConv2Dconv2d_input1sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0*
strides
*
paddingVALID?
*sequential_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_conv2d_biasadd_readvariableop_conv2d_bias*
dtype0*
_output_shapes
:?
sequential_1/conv2d/BiasAddBiasAdd#sequential_1/conv2d/Conv2D:output:02sequential_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
sequential_1/conv2d/ReluRelu$sequential_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
"sequential_1/max_pooling2d/MaxPoolMaxPool&sequential_1/conv2d/Relu:activations:0*/
_output_shapes
:?????????*
strides
*
ksize
*
paddingVALID?
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;sequential_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*
dtype0*&
_output_shapes
:?
sequential_1/conv2d_1/Conv2DConv2D+sequential_1/max_pooling2d/MaxPool:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*/
_output_shapes
:?????????*
T0?
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:sequential_1_conv2d_1_biasadd_readvariableop_conv2d_1_bias*
dtype0*
_output_shapes
:?
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????*
T0?
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:??????????
sequential_1/dropout/IdentityIdentity-sequential_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????p
sequential_1/flatten/ShapeShape&sequential_1/dropout/Identity:output:0*
_output_shapes
:*
T0r
(sequential_1/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:t
*sequential_1/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:t
*sequential_1/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
"sequential_1/flatten/strided_sliceStridedSlice#sequential_1/flatten/Shape:output:01sequential_1/flatten/strided_slice/stack:output:03sequential_1/flatten/strided_slice/stack_1:output:03sequential_1/flatten/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0o
$sequential_1/flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: ?
"sequential_1/flatten/Reshape/shapePack+sequential_1/flatten/strided_slice:output:0-sequential_1/flatten/Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:?
sequential_1/flatten/ReshapeReshape&sequential_1/dropout/Identity:output:0+sequential_1/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp5sequential_1_dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	? ?
sequential_1/dense/MatMulMatMul%sequential_1/flatten/Reshape:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
: ?
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:????????? *
T0?
)sequential_1/activation_1/PartitionedCallPartitionedCall#sequential_1/dense/BiasAdd:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-106*
fR
__inference_isrlu_105*
Tout
2?
sequential_1/dropout_1/IdentityIdentity2sequential_1/activation_1/PartitionedCall:output:0*'
_output_shapes
:????????? *
T0?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp9sequential_1_dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

: 
?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:
?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0+^sequential_1/conv2d/BiasAdd/ReadVariableOp*^sequential_1/conv2d/Conv2D/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/conv2d/BiasAdd/ReadVariableOp*sequential_1/conv2d/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_903

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1319

inputs
identity?Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
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
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:????????? ?
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
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:????????? o
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
?
?
&__inference_dense_1_layer_call_fn_1352

inputs*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_932*
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
**
_gradient_op_typePartitionedCall-939?
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
?	
\
@__inference_flatten_layer_call_and_return_conditional_losses_811

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
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
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
T0*
N*
_output_shapes
:e
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
?
-
__inference_isrlu_1299
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
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_1240

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
?
+__inference_sequential_1_layer_call_fn_1053
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_input%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*
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
*+
_gradient_op_typePartitionedCall-1042*O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_1041?
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
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_1289

inputs
identity?
PartitionedCallPartitionedCallinputs*
fR
__inference_isrlu_105*
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
_gradient_op_typePartitionedCall-106`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_1329

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-907*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_895*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?*
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_952
conv2d_input0
,conv2d_statefulpartitionedcall_conv2d_kernel.
*conv2d_statefulpartitionedcall_conv2d_bias4
0conv2d_1_statefulpartitionedcall_conv2d_1_kernel2
.conv2d_1_statefulpartitionedcall_conv2d_1_bias.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias**
_gradient_op_typePartitionedCall-668*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_661*
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
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-689*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682*
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
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-714*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-735*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-785*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_773*
Tout
2?
flatten/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-818*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_811*
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
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-843*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_836*
Tout
2?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-867*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_860*
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
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall**
_gradient_op_typePartitionedCall-907*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_895*
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
:????????? ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias**
_gradient_op_typePartitionedCall-939*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_932*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2?
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
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : : :, (
&
_user_specified_nameconv2d_input: : : 
?'
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1041

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
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias**
_gradient_op_typePartitionedCall-668*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_661*
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
_gradient_op_typePartitionedCall-689*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682*
Tout
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707*
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
:?????????**
_gradient_op_typePartitionedCall-714?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*-
config_proto

CPU

GPU2*0J 8*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-735*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728*
Tout
2?
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_781*
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
2**
_gradient_op_typePartitionedCall-794?
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
_gradient_op_typePartitionedCall-818*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_811*
Tout
2?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
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
:????????? **
_gradient_op_typePartitionedCall-843*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_836?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-867*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_860*
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
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-916*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_903*
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
:????????? ?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias**
_gradient_op_typePartitionedCall-939*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_932*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2?
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
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
?
_
@__inference_dropout_layer_call_and_return_conditional_losses_773

inputs
identity?Q
dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
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
dtype0*/
_output_shapes
:??????????
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
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
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
?
?
&__inference_conv2d_1_layer_call_fn_719

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
_gradient_op_typePartitionedCall-714*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?

?
?__inference_conv2d_layer_call_and_return_conditional_losses_661

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
?
?
>__inference_dense_layer_call_and_return_conditional_losses_836

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
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?Q
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1145

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel-
)conv2d_biasadd_readvariableop_conv2d_bias2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel1
-conv2d_1_biasadd_readvariableop_conv2d_1_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*
dtype0*&
_output_shapes
:?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:?????????*
T0*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
dtype0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:?????????*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*
dtype0*&
_output_shapes
:?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:??????????
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
dtype0*
_output_shapes
:?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:?????????Y
dropout/dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: e
dropout/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*/
_output_shapes
:?????????*
T0?
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:??????????
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:?????????Z
dropout/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0^
dropout/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*/
_output_shapes
:?????????*
T0?
dropout/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:??????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:?????????*

SrcT0
?
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????V
flatten/ShapeShapedropout/dropout/mul_1:z:0*
T0*
_output_shapes
:e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0b
flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: ?
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:?
flatten/ReshapeReshapedropout/dropout/mul_1:z:0flatten/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	? ?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:????????? *
T0?
activation_1/PartitionedCallPartitionedCalldense/BiasAdd:output:0**
_gradient_op_typePartitionedCall-106*
fR
__inference_isrlu_105*
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
:????????? [
dropout_1/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: l
dropout_1/dropout/ShapeShape%activation_1/PartitionedCall:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:????????? ?
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:????????? ?
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:????????? \
dropout_1/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*'
_output_shapes
:????????? ?
dropout_1/dropout/mulMul%activation_1/PartitionedCall:output:0dropout_1/dropout/truediv:z:0*
T0*'
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:????????? ?
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

: 
?
dense_1/MatMulMatMuldropout_1/dropout/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: : : : : 
?
I
-__inference_max_pooling2d_1_layer_call_fn_738

inputs
identity?
PartitionedCallPartitionedCallinputs*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*J
_output_shapes8
6:4????????????????????????????????????**
_gradient_op_typePartitionedCall-735?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
$__inference_conv2d_layer_call_fn_673

inputs)
%statefulpartitionedcall_conv2d_kernel'
#statefulpartitionedcall_conv2d_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias**
_gradient_op_typePartitionedCall-668*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_661*
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
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?

?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707

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
?
?
+__inference_sequential_1_layer_call_fn_1202

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
StatefulPartitionedCallStatefulPartitionedCallinputs%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*-
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
_gradient_op_typePartitionedCall-1003*O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_1002*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
?
?
$__inference_dense_layer_call_fn_1284

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_836*
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
_gradient_op_typePartitionedCall-843?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1324

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_1250

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-794*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_781*
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
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_1245

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
2**
_gradient_op_typePartitionedCall-785*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_773?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
,
__inference_isrlu_105
x
identityA
SignSignx*'
_output_shapes
:????????? *
T0I
IdentityIdentityx*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :! 

_user_specified_namex
?
?
+__inference_sequential_1_layer_call_fn_1014
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_input%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*+
_gradient_op_typePartitionedCall-1003*O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_1002*
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
?
B
&__inference_flatten_layer_call_fn_1267

inputs
identity?
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-818*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_811*
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
:??????????a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*.
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
?__inference_dense_layer_call_and_return_conditional_losses_1277

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
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
"__inference_signature_wrapper_1068
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_input%statefulpartitionedcall_conv2d_kernel#statefulpartitionedcall_conv2d_bias'statefulpartitionedcall_conv2d_1_kernel%statefulpartitionedcall_conv2d_1_bias$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*
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
*+
_gradient_op_typePartitionedCall-1057*'
f"R 
__inference__wrapped_model_646?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameconv2d_input: : : : : : : 
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_1235

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
?
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728

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
?	
?
@__inference_dense_1_layer_call_and_return_conditional_losses_932

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
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?1
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1189

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel-
)conv2d_biasadd_readvariableop_conv2d_bias2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernel1
-conv2d_1_biasadd_readvariableop_conv2d_1_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*
dtype0*&
_output_shapes
:?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????*
T0*
strides
*
paddingVALID?
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_conv2d_bias*
dtype0*
_output_shapes
:?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:?????????*
T0?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*
dtype0*&
_output_shapes
:?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:??????????
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_1_biasadd_readvariableop_conv2d_1_bias*
dtype0*
_output_shapes
:?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:?????????*
strides
x
dropout/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????V
flatten/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskb
flatten/Reshape/shape/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: ?
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
T0*
N*
_output_shapes
:?
flatten/ReshapeReshapedropout/Identity:output:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	? ?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
dtype0*
_output_shapes
: ?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:????????? *
T0?
activation_1/PartitionedCallPartitionedCalldense/BiasAdd:output:0**
_gradient_op_typePartitionedCall-106*
fR
__inference_isrlu_105*
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
2w
dropout_1/IdentityIdentity%activation_1/PartitionedCall:output:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
dtype0*
_output_shapes

: 
?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitydense_1/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
?
a
E__inference_activation_1_layer_call_and_return_conditional_losses_860

inputs
identity?
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
_gradient_op_typePartitionedCall-106*
fR
__inference_isrlu_105`
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
B__inference_dropout_1_layer_call_and_return_conditional_losses_895

inputs
identity?Q
dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
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
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??b
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
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*'
_output_shapes
:????????? *
T0Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?'
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_977
conv2d_input0
,conv2d_statefulpartitionedcall_conv2d_kernel.
*conv2d_statefulpartitionedcall_conv2d_bias4
0conv2d_1_statefulpartitionedcall_conv2d_1_kernel2
.conv2d_1_statefulpartitionedcall_conv2d_1_bias.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_bias2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_bias
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input,conv2d_statefulpartitionedcall_conv2d_kernel*conv2d_statefulpartitionedcall_conv2d_bias*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_661*
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
_gradient_op_typePartitionedCall-668?
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
_gradient_op_typePartitionedCall-689*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682*
Tout
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:00conv2d_1_statefulpartitionedcall_conv2d_1_kernel.conv2d_1_statefulpartitionedcall_conv2d_1_bias*
Tin
2*/
_output_shapes
:?????????**
_gradient_op_typePartitionedCall-714*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-735*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728*
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
dropout/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-794*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_781*
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
flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_811*
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
:??????????**
_gradient_op_typePartitionedCall-818?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*
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
_gradient_op_typePartitionedCall-843*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_836?
activation_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:????????? **
_gradient_op_typePartitionedCall-867*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_860*
Tout
2*-
config_proto

CPU

GPU2*0J 8?
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-916*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_903*
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
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias**
_gradient_op_typePartitionedCall-939*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_932*
Tout
2*-
config_proto

CPU

GPU2*0J 8*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
?
^
@__inference_dropout_layer_call_and_return_conditional_losses_781

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
?
D
(__inference_dropout_1_layer_call_fn_1334

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
_gradient_op_typePartitionedCall-916*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_903`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:????????? *
T0"
identityIdentity:output:0*&
_input_shapes
:????????? :& "
 
_user_specified_nameinputs"?L
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
?;
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?7
_tf_keras_sequential?7{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "input_spec": null, "activity_regularizer": null}
?

kernel
bias
_callable_losses
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "filters": 8, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "activity_regularizer": null}
?
_callable_losses
regularization_losses
trainable_variables
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
?

"kernel
#bias
$_callable_losses
%regularization_losses
&trainable_variables
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "activity_regularizer": null}
?
)_callable_losses
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "activity_regularizer": null}
?
._callable_losses
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "input_spec": null, "activity_regularizer": null}
?
3_callable_losses
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "activity_regularizer": null}
?

8kernel
9bias
:_callable_losses
;regularization_losses
<trainable_variables
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "activity_regularizer": null}
?
?_callable_losses
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__
?
activation"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "isrlu"}, "input_spec": null, "activity_regularizer": null}
?
D_callable_losses
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "input_spec": null, "activity_regularizer": null}
?

Ikernel
Jbias
K_callable_losses
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "activity_regularizer": null}
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem?m?"m?#m?8m?9m?Im?Jm?v?v?"v?#v?8v?9v?Iv?Jv?"
	optimizer
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
regularization_losses
Umetrics
trainable_variables
	variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Ymetrics
trainable_variables
	variables
Zlayer_regularization_losses

[layers
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
]metrics
trainable_variables
	variables
^layer_regularization_losses

_layers
`non_trainable_variables
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
regularization_losses
ametrics
trainable_variables
 	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
%regularization_losses
emetrics
&trainable_variables
'	variables
flayer_regularization_losses

glayers
hnon_trainable_variables
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
*regularization_losses
imetrics
+trainable_variables
,	variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
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
/regularization_losses
mmetrics
0trainable_variables
1	variables
nlayer_regularization_losses

olayers
pnon_trainable_variables
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
4regularization_losses
qmetrics
5trainable_variables
6	variables
rlayer_regularization_losses

slayers
tnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	? 2dense/kernel
: 2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
;regularization_losses
umetrics
<trainable_variables
=	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
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
@regularization_losses
ymetrics
Atrainable_variables
B	variables
zlayer_regularization_losses

{layers
|non_trainable_variables
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
Eregularization_losses
}metrics
Ftrainable_variables
G	variables
~layer_regularization_losses

layers
?non_trainable_variables
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
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
Lregularization_losses
?metrics
Mtrainable_variables
N	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
(
?0"
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
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?_updates
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "acc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?metrics
?trainable_variables
?	variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
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
0
?0
?1"
trackable_list_wrapper
5:32training/Adam/conv2d/kernel/m
':%2training/Adam/conv2d/bias/m
7:52training/Adam/conv2d_1/kernel/m
):'2training/Adam/conv2d_1/bias/m
-:+	? 2training/Adam/dense/kernel/m
&:$ 2training/Adam/dense/bias/m
.:, 
2training/Adam/dense_1/kernel/m
(:&
2training/Adam/dense_1/bias/m
5:32training/Adam/conv2d/kernel/v
':%2training/Adam/conv2d/bias/v
7:52training/Adam/conv2d_1/kernel/v
):'2training/Adam/conv2d_1/bias/v
-:+	? 2training/Adam/dense/kernel/v
&:$ 2training/Adam/dense/bias/v
.:, 
2training/Adam/dense_1/kernel/v
(:&
2training/Adam/dense_1/bias/v
?2?
__inference__wrapped_model_646?
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
?2?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1145
E__inference_sequential_1_layer_call_and_return_conditional_losses_952
E__inference_sequential_1_layer_call_and_return_conditional_losses_977
F__inference_sequential_1_layer_call_and_return_conditional_losses_1189?
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
?2?
+__inference_sequential_1_layer_call_fn_1202
+__inference_sequential_1_layer_call_fn_1053
+__inference_sequential_1_layer_call_fn_1215
+__inference_sequential_1_layer_call_fn_1014?
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
?2?
?__inference_conv2d_layer_call_and_return_conditional_losses_661?
???
FullArgSpec
args?
jself
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
annotations? *7?4
2?/+???????????????????????????
?2?
$__inference_conv2d_layer_call_fn_673?
???
FullArgSpec
args?
jself
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
annotations? *7?4
2?/+???????????????????????????
?2?
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682?
???
FullArgSpec
args?
jself
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
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_max_pooling2d_layer_call_fn_692?
???
FullArgSpec
args?
jself
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
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707?
???
FullArgSpec
args?
jself
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
annotations? *7?4
2?/+???????????????????????????
?2?
&__inference_conv2d_1_layer_call_fn_719?
???
FullArgSpec
args?
jself
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
annotations? *7?4
2?/+???????????????????????????
?2?
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728?
???
FullArgSpec
args?
jself
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
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_1_layer_call_fn_738?
???
FullArgSpec
args?
jself
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
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_1235
A__inference_dropout_layer_call_and_return_conditional_losses_1240?
???
FullArgSpec)
args!?
jself
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
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_1250
&__inference_dropout_layer_call_fn_1245?
???
FullArgSpec)
args!?
jself
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
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_1262?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_1267?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_1277?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
$__inference_dense_layer_call_fn_1284?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
F__inference_activation_1_layer_call_and_return_conditional_losses_1289?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
+__inference_activation_1_layer_call_fn_1294?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
__inference_isrlu_1299?
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
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_1324
C__inference_dropout_1_layer_call_and_return_conditional_losses_1319?
???
FullArgSpec)
args!?
jself
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
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_1329
(__inference_dropout_1_layer_call_fn_1334?
???
FullArgSpec)
args!?
jself
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
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_1345?
???
FullArgSpec
args?
jself
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
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_1352?
???
FullArgSpec
args?
jself
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
annotations? *
 
6B4
"__inference_signature_wrapper_1068conv2d_input
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
?__inference_conv2d_layer_call_and_return_conditional_losses_661?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_728?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? x
$__inference_dense_layer_call_fn_1284P890?-
&?#
!?
inputs??????????
? "?????????? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1189r"#89IJ??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
+__inference_sequential_1_layer_call_fn_1053k"#89IJE?B
;?8
.?+
conv2d_input?????????
p 

 
? "??????????
?
C__inference_dropout_1_layer_call_and_return_conditional_losses_1319\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_1324\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
$__inference_conv2d_layer_call_fn_673?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
A__inference_dense_1_layer_call_and_return_conditional_losses_1345\IJ/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????

? ?
&__inference_dropout_layer_call_fn_1245_;?8
1?.
(?%
inputs?????????
p
? " ???????????
&__inference_dropout_layer_call_fn_1250_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
E__inference_sequential_1_layer_call_and_return_conditional_losses_952x"#89IJE?B
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
? ?
__inference__wrapped_model_646|"#89IJ=?:
3?0
.?+
conv2d_input?????????
? "1?.
,
dense_1!?
dense_1?????????
~
&__inference_flatten_layer_call_fn_1267T7?4
-?*
(?%
inputs?????????
? "????????????
+__inference_sequential_1_layer_call_fn_1202e"#89IJ??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
+__inference_max_pooling2d_layer_call_fn_692?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_sequential_1_layer_call_and_return_conditional_losses_977x"#89IJE?B
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
"__inference_signature_wrapper_1068?"#89IJM?J
? 
C?@
>
conv2d_input.?+
conv2d_input?????????"1?.
,
dense_1!?
dense_1?????????
?
+__inference_sequential_1_layer_call_fn_1215e"#89IJ??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
A__inference_flatten_layer_call_and_return_conditional_losses_1262a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? z
+__inference_activation_1_layer_call_fn_1294K/?,
%?"
 ?
inputs????????? 
? "?????????? y
&__inference_dense_1_layer_call_fn_1352OIJ/?,
%?"
 ?
inputs????????? 
? "??????????
?
F__inference_activation_1_layer_call_and_return_conditional_losses_1289X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1145r"#89IJ??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
?__inference_dense_layer_call_and_return_conditional_losses_1277]890?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_1240l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_1235l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
+__inference_sequential_1_layer_call_fn_1014k"#89IJE?B
;?8
.?+
conv2d_input?????????
p

 
? "??????????
{
(__inference_dropout_1_layer_call_fn_1329O3?0
)?&
 ?
inputs????????? 
p
? "?????????? {
(__inference_dropout_1_layer_call_fn_1334O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
-__inference_max_pooling2d_1_layer_call_fn_738?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_conv2d_1_layer_call_fn_719?"#I?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????d
__inference_isrlu_1299J.?+
$?!
?
x????????? 
`
? "?????????? ?
A__inference_conv2d_1_layer_call_and_return_conditional_losses_707?"#I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_682?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? 