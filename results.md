
## 2026-10-15 16:39 | baseline

Model: `LogisticRegression(max_iter=1000)`

```
                precision    recall  f1-score   support

          ARTS       0.66      0.37      0.47       779
      BUSINESS       0.64      0.47      0.54      1546
         CRIME       0.71      0.55      0.62       711
     EDUCATION       0.59      0.31      0.40       432
 ENTERTAINMENT       0.68      0.76      0.72      4533
   ENVIRONMENT       0.57      0.38      0.46       797
        FAMILY       0.75      0.73      0.74      3925
          FOOD       0.80      0.72      0.76      1685
     LIFESTYLE       0.78      0.77      0.77      3156
      POLITICS       0.76      0.86      0.81      7096
      RELIGION       0.69      0.35      0.47       513
SCIENCE & TECH       0.68      0.40      0.51       850
        SPORTS       0.78      0.58      0.67      1015
        TRAVEL       0.74      0.71      0.73      1975
      WELLNESS       0.64      0.81      0.71      4912
         WORLD       0.74      0.66      0.70      1906

      accuracy                           0.72     35831
     macro avg       0.70      0.59      0.63     35831
  weighted avg       0.72      0.72      0.71     35831
```

## 2026-10-15 16:40 | class_weight balanced

Model: `LogisticRegression(class_weight='balanced', max_iter=1000)`

```
                precision    recall  f1-score   support

          ARTS       0.38      0.59      0.47       779
      BUSINESS       0.50      0.61      0.55      1546
         CRIME       0.47      0.74      0.57       711
     EDUCATION       0.31      0.62      0.41       432
 ENTERTAINMENT       0.76      0.66      0.71      4533
   ENVIRONMENT       0.42      0.68      0.52       797
        FAMILY       0.78      0.69      0.73      3925
          FOOD       0.71      0.80      0.75      1685
     LIFESTYLE       0.80      0.75      0.77      3156
      POLITICS       0.89      0.70      0.79      7096
      RELIGION       0.43      0.65      0.52       513
SCIENCE & TECH       0.42      0.59      0.49       850
        SPORTS       0.64      0.76      0.69      1015
        TRAVEL       0.69      0.74      0.72      1975
      WELLNESS       0.75      0.66      0.70      4912
         WORLD       0.68      0.72      0.70      1906

      accuracy                           0.69     35831
     macro avg       0.60      0.68      0.63     35831
  weighted avg       0.73      0.69      0.70     35831
```

## 2026-10-15 16:45 | class_weight balanced

Model: `LogisticRegression(class_weight='balanced', max_iter=1000)`

```
                precision    recall  f1-score   support

          ARTS       0.38      0.59      0.47       779
      BUSINESS       0.50      0.61      0.55      1546
         CRIME       0.47      0.74      0.57       711
     EDUCATION       0.31      0.62      0.41       432
 ENTERTAINMENT       0.76      0.66      0.71      4533
   ENVIRONMENT       0.42      0.68      0.52       797
        FAMILY       0.78      0.69      0.73      3925
          FOOD       0.71      0.80      0.75      1685
     LIFESTYLE       0.80      0.75      0.77      3156
      POLITICS       0.89      0.70      0.79      7096
      RELIGION       0.43      0.65      0.52       513
SCIENCE & TECH       0.42      0.59      0.49       850
        SPORTS       0.64      0.76      0.69      1015
        TRAVEL       0.69      0.74      0.72      1975
      WELLNESS       0.75      0.66      0.70      4912
         WORLD       0.68      0.72      0.70      1906

      accuracy                           0.69     35831
     macro avg       0.60      0.68      0.63     35831
  weighted avg       0.73      0.69      0.70     35831
```
