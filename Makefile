random-forest: main.c src/Utils.c src/TrainingForest.c src/Predictions.c
	gcc -o random-forest main.c src/Utils.c src/TrainingForest.c src/Predictions.c