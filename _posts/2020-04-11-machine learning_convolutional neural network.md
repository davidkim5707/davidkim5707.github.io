---
layout: post
title: machine learning_convolutional neural network
categories:
  - Programming
tags:
  - Python
  - convolutional neural network
last_modified_at: 2020-04-11
use_math: true
---
source& copyright: lecture note in DataScienceLab in Yonsei university  

### DenseNetwork 


```python
import tensorflow as tf
import numpy as np
```

### ![image.png](attachment:image.png)

### ![image.png](attachment:image.png)

## 하이퍼 파라미터


```python
EPOCHS = 10
```

## DenseUnit 구현


```python
class DenseUnit(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(DenseUnit, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False, mask=None): # x: (Batch, H, W, Ch_in)
        # 여기서도 Res Net처럼 "pre-activation" 구조 ( BN - ReLU - Conv 순 )
        h = self.bn(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv(h) # h: (Batch, H, W, filter_output)
        return self.concat([x, h]) # (Batch, H, W, (Ch_in + filter_output))
```

## DenseLayer 구현


```python
class DenseLayer(tf.keras.Model):    
    def __init__(self, num_unit, growth_rate, kernel_size):
        super(DenseLayer, self).__init__()
        self.sequence = list()
        for idx in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x
```

## Transition Layer 구현

- conv & pooling


```python
class TransitionLayer(tf.keras.Model):    
    def __init__(self, filters, kernel_size):
        super(TransitionLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, x, training=False, mask=None):
        x = self.conv(x)
        return self.pool(x)
```

## 모델 정의


```python
class DenseNet(tf.keras.Model):
    
    def __init__(self):
        super(DenseNet, self).__init__()   # ex. 28*28*n의 input
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu') # 28x28x8
        
        self.dl1 = DenseLayer(2, 4, (3, 3)) # 28x28x16 ( = 8개 filter +(2개 unit *4의 growth rate) )
        self.tr1 = TransitionLayer(16, (3, 3)) # 14x14x16 ( max pooling )
        
        self.dl2 = DenseLayer(2, 8, (3, 3)) # 14x14x32 ( = 16개 filter + (2개 unit*8의 growth rate) )
        self.tr2 = TransitionLayer(32, (3, 3)) # 7x7x32 ( max pooling )
        
        self.dl3 = DenseLayer(2, 16, (3, 3)) # 7x7x64 ( = 32개 filter + (2개 unit*16의 growth rate))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')       

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        
        x = self.dl1(x, training=training)
        x = self.tr1(x)

        x = self.dl2(x, training=training)
        x = self.tr2(x)
        
        x = self.dl3(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
        
```

## 학습, 테스트 루프 정의


```python
# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

## 데이터셋 준비



```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

## 학습 환경 정의
### 모델 생성, 손실함수, 최적화 알고리즘, 평가지표 정의


```python
# Create model
model = DenseNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

## 학습 루프 동작


```python
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```

    Epoch 1, Loss: 0.11222333461046219, Accuracy: 96.69999694824219, Test Loss: 0.051826879382133484, Test Accuracy: 98.29999542236328
    Epoch 2, Loss: 0.08431833237409592, Accuracy: 97.55833435058594, Test Loss: 0.052275653928518295, Test Accuracy: 98.3699951171875
    Epoch 3, Loss: 0.07055088877677917, Accuracy: 97.97944641113281, Test Loss: 0.059118326753377914, Test Accuracy: 98.28666687011719
    Epoch 4, Loss: 0.06249263882637024, Accuracy: 98.22666931152344, Test Loss: 0.06738369911909103, Test Accuracy: 98.18999481201172
    Epoch 5, Loss: 0.0574798583984375, Accuracy: 98.3846664428711, Test Loss: 0.06918792426586151, Test Accuracy: 98.2280044555664
    Epoch 6, Loss: 0.05322617292404175, Accuracy: 98.51972198486328, Test Loss: 0.06587052345275879, Test Accuracy: 98.32833099365234
    Epoch 7, Loss: 0.049832653254270554, Accuracy: 98.62833404541016, Test Loss: 0.06826455891132355, Test Accuracy: 98.36714172363281
    Epoch 8, Loss: 0.04693884775042534, Accuracy: 98.71583557128906, Test Loss: 0.06756298989057541, Test Accuracy: 98.42124938964844
    Epoch 9, Loss: 0.04515141248703003, Accuracy: 98.77925872802734, Test Loss: 0.07314397394657135, Test Accuracy: 98.36333465576172
    Epoch 10, Loss: 0.04279949143528938, Accuracy: 98.84683227539062, Test Loss: 0.07374981045722961, Test Accuracy: 98.39399719238281
    
### Residual Network

>### Residual Unit -> Residual Layer -> Residual Net


```python
import tensorflow as tf
import numpy as np
```

## 하이퍼 파라미터


```python
EPOCHS = 10
```

## Residual Unit 구현

![image.png](attachment:image.png)


```python
class ResidualUnit(tf.keras.Model):
    
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        
        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding='same')

    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)
        
        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h
```

## Residual Layer 구현

![image.png](attachment:image.png)


```python
class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()
        for f_in, f_out in zip([filter_in] + list(filters), filters): # ex. zip([8,16,16],[16,16])
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x
```

## 모델 정의


```python
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu') # 28x28x8
        
        self.res1 = ResnetLayer(8, (16, 16), (3, 3)) # 28x28x16
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2)) # 14x14x16
        
        self.res2 = ResnetLayer(16, (32, 32), (3, 3)) # 14x14x32
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2)) # 7x7x32
        
        self.res3 = ResnetLayer(32, (64, 64), (3, 3)) # 7x7x64
        
        self.flatten = tf.keras.layers.Flatten() # 3136
        self.dense1 = tf.keras.layers.Dense(128, activation='relu') # 128
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax') # 10 ( classify to number 0 ~ 9)
        
    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        
        x = self.res1(x, training=training)
        x = self.pool1(x)
        x = self.res2(x, training=training)
        x = self.pool2(x)
        x = self.res3(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
        
```

## 학습, 테스트 루프 정의


```python
# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

## 데이터셋 준비



```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

## 학습 환경 정의
### 모델 생성, 손실함수, 최적화 알고리즘, 평가지표 정의


```python
# Create model
model = ResNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

## 학습 루프 동작


```python
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```

    Epoch 1, Loss: 0.1362374722957611, Accuracy: 96.26166534423828, Test Loss: 0.043587375432252884, Test Accuracy: 98.6500015258789
    Epoch 2, Loss: 0.09931698441505432, Accuracy: 97.2874984741211, Test Loss: 0.046643007546663284, Test Accuracy: 98.68999481201172
    Epoch 3, Loss: 0.0823485255241394, Accuracy: 97.7411117553711, Test Loss: 0.044813815504312515, Test Accuracy: 98.79666900634766
    Epoch 4, Loss: 0.07157806307077408, Accuracy: 98.02749633789062, Test Loss: 0.04259391874074936, Test Accuracy: 98.84249877929688
    Epoch 5, Loss: 0.06405625492334366, Accuracy: 98.22633361816406, Test Loss: 0.04102792590856552, Test Accuracy: 98.85600280761719
    Epoch 6, Loss: 0.05831533297896385, Accuracy: 98.37750244140625, Test Loss: 0.04298485442996025, Test Accuracy: 98.81666564941406
    Epoch 7, Loss: 0.05337631702423096, Accuracy: 98.5088119506836, Test Loss: 0.04212458059191704, Test Accuracy: 98.83428955078125
    Epoch 8, Loss: 0.04950246587395668, Accuracy: 98.61895751953125, Test Loss: 0.04147474840283394, Test Accuracy: 98.85625457763672
    Epoch 9, Loss: 0.04653759300708771, Accuracy: 98.70240783691406, Test Loss: 0.040949009358882904, Test Accuracy: 98.87777709960938
    Epoch 10, Loss: 0.044119786471128464, Accuracy: 98.77149963378906, Test Loss: 0.04024982452392578, Test Accuracy: 98.91099548339844
    
###basic convoluted neural network


```python
import tensorflow as tf
import numpy as np
```

## 하이퍼 파라미터


```python
EPOCHS = 10
```

## 모델 정의


```python
class ConvNet(tf.keras.Model):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # layer을 많이 쌓을거면 다음과 같이하면 편리!
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.MaxPool2D
        
        self.sequence = list()
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16
        self.sequence.append(conv2d(16, (3, 3), padding='same', activation='relu')) # 28x28x16
        self.sequence.append(maxpool((2,2))) # 14x14x16
        
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(conv2d(32, (3, 3), padding='same', activation='relu')) # 14x14x32
        self.sequence.append(maxpool((2,2))) # 7x7x32
        
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        self.sequence.append(conv2d(64, (3, 3), padding='same', activation='relu')) # 7x7x64
        
        self.sequence.append(tf.keras.layers.Flatten()) # 1568
        self.sequence.append(tf.keras.layers.Dense(128, activation='relu'))
        self.sequence.append(tf.keras.layers.Dense(10, activation='softmax'))

    def call(self, x, training=False, mask=None):
        for layer in self.sequence:
            x = layer(x)
        return x
```

### 구현 연습
- 아래의 모델 구현해보기!

![image.png](attachment:image.png)


```python
class Conv2(tf.keras.Model):
    def __init__(self):
        super(Conv2,self)._init()
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.Maxpool2D
        flatten = tf.keras.layers.Flatten
        dense = tf.keras.layers.Dense
        
        # 28*28*3의 input
        self.sequence = list()
        self.sequence.append(conv2d(32,(5,5),padding='same',activation='relu'))
        self.sequence.append(maxpool((2,2)))
        self.sequence.append(conv2d(32,(5,5),padding='same',activation='relu'))
        self.sequence.append(maxpool((2,2)))
        self.sequence.append(flatten())
        self.sequence.append(dense(128,activation='relu'))
        self.sequence.append(dense(10,activation='softmax'))
    
    def call(self,x,training=False,mask=None):
        for layer in self.sequence:
            x=layer(x)
        return(x)
```

## 학습, 테스트 루프 정의


```python
# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = loss_object(labels, pred)
    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
```

## 데이터셋 준비



```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize

# x_train : (NUM_SAMPLE, 28, 28) -> (NUM_SAMPLE, 28, 28, 1)
# 4차원으로 받아주기 때문에 마지막에 axis 추가

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```

## 학습 환경 정의
### 모델 생성, 손실함수, 최적화 알고리즘, 평가지표 정의


```python
# Create model
#model = ConvNet()
model = ConvSH()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

## 학습 루프 동작


```python
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-4fb3dfc37511> in <module>
          1 for epoch in range(EPOCHS):
          2     for images, labels in train_ds:
    ----> 3         train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
          4 
          5     for test_images, test_labels in test_ds:
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\def_function.py in __call__(self, *args, **kwds)
        455 
        456     tracing_count = self._get_tracing_count()
    --> 457     result = self._call(*args, **kwds)
        458     if tracing_count == self._get_tracing_count():
        459       self._call_counter.called_without_tracing()
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\def_function.py in _call(self, *args, **kwds)
        485       # In this case we have created variables on the first call, so we run the
        486       # defunned version which is guaranteed to never create variables.
    --> 487       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        488     elif self._stateful_fn is not None:
        489       # Release the lock early so that multiple threads can perform the call
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in __call__(self, *args, **kwargs)
       1821     """Calls a graph function specialized to the inputs."""
       1822     graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 1823     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       1824 
       1825   @property
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in _filtered_call(self, args, kwargs)
       1139          if isinstance(t, (ops.Tensor,
       1140                            resource_variable_ops.BaseResourceVariable))),
    -> 1141         self.captured_inputs)
       1142 
       1143   def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1222     if executing_eagerly:
       1223       flat_outputs = forward_function.call(
    -> 1224           ctx, args, cancellation_manager=cancellation_manager)
       1225     else:
       1226       gradient_name = self._delayed_rewrite_functions.register()
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\function.py in call(self, ctx, args, cancellation_manager)
        509               inputs=args,
        510               attrs=("executor_type", executor_type, "config_proto", config),
    --> 511               ctx=ctx)
        512         else:
        513           outputs = execute.execute_with_cancellation(
    

    ~\Anaconda3\lib\site-packages\tensorflow_core\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         59     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,
         60                                                op_name, inputs, attrs,
    ---> 61                                                num_outputs)
         62   except core._NotOkStatusException as e:
         63     if name is not None:
    

    KeyboardInterrupt: 



```python

```
