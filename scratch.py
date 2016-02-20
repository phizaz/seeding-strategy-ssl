testdata

begin()
    .x(points)
    .y(labels)
    .pipe(split_cross(10)) --> (use 'x' and/or 'y') into N pipes with N xs and ys
        .pipe(svm())
        .pipe(result())
    .pipe(merge_average()) -->
    .end()


begin()
    .x(train_points)
    .y(train_labels)
    .pipe(svm()) --> (use 'x' and 'y') mark 'model'
    .x(test_points)
    .pipe(predict()) --> (use 'model' and 'x') mark 'prediction'
    .y(test_labels)
    .pipe(evaluate()) --> (use 'prediction' and 'y') mark 'evaluation'


begin()
    .x(points)
    .y(labels)
    .split(5, split_parallel())
        .y(random_getter('y', 0.1))
        .pipe(kmeans(K=10, times=5)) --> (use 'x') mark 'model'
        .y(consensus()) --> (use 'model' and 'y') mark 'y'
        .pipe(knn(K=3)) --> (use 'x' and 'y') mark 'model'
        .pipe(split_cross(5))
            .pipe(predict()) --> (use 'x' and 'model') mark 'prediction'
            .pipe(evaluate()) --> (use 'prediction' and 'y') mark 'evaluation'
        .pipe(merge(average())) --> (use 'evaluation') mark 'evaluation'
    .merge(merge(average())) --> (use 'evaluation') mark 'evaluation'
