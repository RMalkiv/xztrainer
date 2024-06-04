# Trainable

## Some type aliases

::: xztrainer.model.ModelOutputType
::: xztrainer.model.ModelOutputsType
::: xztrainer.model.DataType

## Context Objects

Context objects are used to interact with various training context such as optimizer, model or scheduler.

::: xztrainer.context

## Trainable Object

Model behaviour on forward pass, metric computation and callbacks are configured by subclassing a `XZTrainable`
class and implementing its functions.

::: xztrainer.trainable.XZTrainable