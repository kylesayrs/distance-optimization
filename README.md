# distance-optimization
Given a list of points and target distances between them, this algorithm calculates a gradient with respect to MSE loss and iteratively moves points closer to their optimal positions. The next point to optimize is decided stochastically and is weighted towards points with a large error value.

<img src="assets/resorts_hotels_218.34.png" alt="Sample Results" width="640" height="480px" />

``` python
minimum_loss = 0.0
max_steps = 1000
learning_rate = 0.5
points = [
    Point([None] * 6 + _negate_values([136, 74, 30, 156, 72, 109, 42, 57]), name="Jumbo Kingdom"),
    Point([None] * 6 + _negate_values([75, 88, 22, 70, 106, 118, 42, 62]), name="World's Fair"),
    Point([None] * 6 + _negate_values([67, 103, 30, 83, 109, 78, 48, 43]), name="Jumbo Studios"),
    Point([None] * 6 + _negate_values([48, 44, 35, 70, 42, 52, 25, 18]), name="Animal Planet Zoo"),
    Point([None] * 6 + _negate_values([32, 44, 47, 43, 48, 23, 19, 16]), name="Trunk Water Park"),
    Point([None] * 6 + _negate_values([27, 17, 3, 17, 15, 18, 56, 32]), name="Jumbo Golf Course"),

    Point(_negate_values([136, 75, 67, 48, 32, 27]) + [None] * 8, name="Tusk Hotel"),
    Point(_negate_values([74, 88, 103, 44, 44, 17]) + [None] * 8, name="Mammoth Motel"),
    Point(_negate_values([30, 22, 30, 35, 47, 3]) + [None] * 8, name="Elephant Lodge"),
    Point(_negate_values([156, 70, 83, 70, 43, 17]) + [None] * 8, name="Trunk Inn"),
    Point(_negate_values([72, 106, 109, 42, 48, 15]) + [None] * 8, name="Loxodon Lodge"),
    Point(_negate_values([109, 118, 78, 52, 23, 18]) + [None] * 8, name="Pachyderm Suites"),
    Point(_negate_values([42, 42, 48, 25, 19, 56]) + [None] * 8, name="Mouse Resort"),
    Point(_negate_values([57, 62, 43, 18, 16, 32]) + [None] * 8, name="Oliphant Camp"),
]

validate_points(points)

optimize_points(
    points,
    learning_rate=learning_rate,
    max_steps=max_steps,
    minimum_loss=minimum_loss
)

plot_points(points)
```

```
point: Point(name="World's Fair", (410.36, 66.82)) | loss: 68520.921083 | total_loss: 25579.05
point: Point(name="World's Fair", (375.98, 121.62)) | loss: 43549.92 | total_loss: 21417.22
point: Point(name="Elephant Lodge", (461.35, 79.23)) | loss: 47633.285950 | total_loss: 21417.22
point: Point(name="Elephant Lodge", (431.25, 122.89)) | loss: 31259.52 | total_loss: 19370.50
point: Point(name="World's Fair", (375.98, 121.62)) | loss: 44577.617669 | total_loss: 19370.50
point: Point(name="World's Fair", (349.98, 164.19)) | loss: 29750.54 | total_loss: 16899.32
point: Point(name="Elephant Lodge", (431.25, 122.89)) | loss: 30006.352409 | total_loss: 16899.32
point: Point(name="Elephant Lodge", (408.94, 156.03)) | loss: 20710.60 | total_loss: 15737.35
point: Point(name="World's Fair", (349.98, 164.19)) | loss: 30562.724003 | total_loss: 15737.35
point: Point(name="World's Fair", (330.38, 197.74)) | loss: 21558.14 | total_loss: 14236.58
point: Point(name="Mammoth Motel", (472.52, 468.63)) | loss: 23911.757913 | total_loss: 14236.58
point: Point(name="Mammoth Motel", (439.69, 448.41)) | loss: 15289.84 | total_loss: 13158.85
point: Point(name="Tusk Hotel", (55.89, 205.41)) | loss: 23631.542479 | total_loss: 13158.85
...
point: Point(name="Trunk Inn", (276.12, 286.89)) | loss: 526.46 | total_loss: 218.35
point: Point(name="Jumbo Golf Course", (268.94, 140.55)) | loss: 283.181844 | total_loss: 218.35
point: Point(name="Jumbo Golf Course", (268.89, 140.45)) | loss: 283.14 | total_loss: 218.34
point: Point(name="Trunk Inn", (276.12, 286.89)) | loss: 525.278451 | total_loss: 218.34
point: Point(name="Trunk Inn", (276.14, 286.88)) | loss: 525.28 | total_loss: 218.34
point: Point(name="Jumbo Studios", (361.07, 357.63)) | loss: 112.406307 | total_loss: 218.34
point: Point(name="Jumbo Studios", (361.10, 357.70)) | loss: 112.39 | total_loss: 218.34
point: Point(name="Jumbo Kingdom", (319.43, 219.89)) | loss: 556.761713 | total_loss: 218.34
point: Point(name="Jumbo Kingdom", (319.43, 219.86)) | loss: 556.76 | total_loss: 218.34
point: Point(name="Trunk Inn", (276.14, 286.88)) | loss: 525.441486 | total_loss: 218.34
point: Point(name="Trunk Inn", (276.15, 286.87)) | loss: 525.44 | total_loss: 218.34
```
