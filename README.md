# distance-optimization
Implements iterative point movement

``` python
minimum_loss = 0.000001
points = [
    Point([-1], [None, None, 0]),
    Point([7], [None, None, 0]),
    Point([1], [None, None, None]),
]

loss = MSELoss(points)
optimizer = SimpleOptimizer(learning_rate=0.5)
```

```
points: [Point (-1.00), Point (4.00), Point (1.00)] | loss: 4.33
points: [Point (-1.00), Point (2.50), Point (1.00)] | loss: 2.08
points: [Point (0.00), Point (2.50), Point (1.00)] | loss: 1.08
points: [Point (0.00), Point (1.75), Point (1.00)] | loss: 0.52
points: [Point (0.50), Point (1.75), Point (1.00)] | loss: 0.27
points: [Point (0.50), Point (1.38), Point (1.00)] | loss: 0.13
points: [Point (0.75), Point (1.38), Point (1.00)] | loss: 0.07
points: [Point (0.75), Point (1.19), Point (1.00)] | loss: 0.03
points: [Point (0.88), Point (1.19), Point (1.00)] | loss: 0.02
points: [Point (0.88), Point (1.09), Point (1.00)] | loss: 0.01
points: [Point (0.94), Point (1.09), Point (1.00)] | loss: 0.00
points: [Point (0.94), Point (1.05), Point (1.00)] | loss: 0.00
points: [Point (0.97), Point (1.05), Point (1.00)] | loss: 0.00
points: [Point (0.97), Point (1.02), Point (1.00)] | loss: 0.00
points: [Point (0.98), Point (1.02), Point (1.00)] | loss: 0.00
points: [Point (0.98), Point (1.01), Point (1.00)] | loss: 0.00
points: [Point (0.99), Point (1.01), Point (1.00)] | loss: 0.00
points: [Point (0.99), Point (1.01), Point (1.00)] | loss: 0.00
points: [Point (1.00), Point (1.01), Point (1.00)] | loss: 0.00
points: [Point (1.00), Point (1.00), Point (1.00)] | loss: 0.00
```
