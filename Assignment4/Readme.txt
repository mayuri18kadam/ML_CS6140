Please open the given project folder as it is in your IDE.
You may want to install certain plugins to run the programs:
numpy,
pandas,
pip,
scikit-learn,
scipy,
sklearn,
matplotlib

Q1)

Q1.2.1 - Refer to Question1.2.1.py for implementation of perceptron algo
	Q1.2.1_SelectingParams.txt and Q1.2.1_AccuracyVSLearningRate.png were used to determine the best value of learning rate.
	Q1.2.1_FinalOutput - final output using the best learning rate = 0.00005, Test Accuracy Mean =  0.9970000000000001

Q1.2.2 - Refer to Question1.2.2.py for implementation of dual perceptron
	Q1.2.2_FinalOutput - final output for the dual perceptron; Test Accuracy Mean =  0.9890000000000001

Q1.2.3 - for perceptron algo : Test Accuracy Mean =  0.9970000000000001
	for dual : Test Accuracy Mean =  0.9890000000000001
	Hence, we can observe that the values are almost identical

Q1.3.1 - Refer to Question1.3.1.py for implementation of dual perceptron with linear kernel
	Q1.3.1_FinalOutput.txt - final output for dual perceptron with linear kernel; Test Accuracy Mean =  0.5

Q1.3.1 - Refer to Question1.3.2.py for implementation of dual perceptron with RBF kernel
	Q1.3.2_SelectingParams.txt and Q1.3.2_AccuracyVSGamma.png were used to determine the best value of gamma
	Q1.3.2_FinalOutput.txt - final output using the best gamma = 0.15; Test Accuracy Mean =  0.592

****************************************************************************************************************************************

Q2)

Q2.1 - Yes we need w0 in regularization as well as that is the bias (acts as intercept) of original hyperplane

Q2.2 - refer Assignment4 q2 theory.pdf

Q2.3 - Cancer - best combination of params: learningRate, tol, lam = 0.005, 0.0005, 0.5 respectively 
	Diabetes - best combination of params: learningRate, tol, lam = 0.005, 0.005, 5 respectively 
	Spambase - best combination of params: learningRate, tol, lam = 0.00005, 0.005, 5 respectively 
	
	refer to Question2.py for an implementation of the objective function given in the assignment
	best values:
	Cancer: 
	Diabetes:
	Spambase:
	refer to Question2New.py for an implementation of the logistic regression as in the previous assignment with the regularization paramter, lambda
	best values:
	Cancer: 
	Diabetes:
	Spambase:

Q2.4 - Values got using Logistic regression without regularization:
	Cancer: Test Accuracy Mean =  0.9524749373433584
	Diabetes: Test Accuracy Mean =  0.7486329460013671
	Spambase: Test Accuracy Mean =  0.8934966518909743
	
	As can be observed 

Q2.5 - 

**************************************************************************************************************************************

Q3)

Q3.1 - refer to Question3.1.1.py for implementation
	refer to Q3.1.1_Cancer.txt, Q3.1.1_Diabetes.txt for detail output
	Cancer:
		best C =  128.0
		best Gamma =  0.03125
		SVM Test Accuracy Mean =  0.9683897243107771
		RBF Test Accuracy Mean =  0.9245300751879698
	Diabetes:
		best C =  0.25
		best Gamma =  0.015625
		SVM Test Accuracy Mean =  0.7721462747778537
		RBF Test Accuracy Mean =  0.7343984962406016

Q3.2 - refer to Question3.1.2.py for implementation
	Cancer:
		best C =  64.0
		best Gamma =  0.0009765625
		Test Accuracy Mean =  0.9594915521759599
		Test Accuracy Mean =  0.9415291547220894
	Diabetes:
		best C =  256.0
		best Gamma =  0.03125
		SVM Test Accuracy Mean =  0.7226043503302518
		RBF Test Accuracy Mean =  0.6359945681364391

Q3.3 - refer:
	Cancer ROC-AUC curve across all k-folds: Q3.1.2_Cancer_Linear_ROCAUC.png, Q3.1.2_Cancer_RBF_ROCAUC.png
	Diabetes ROC-AUC curve across all k-folds: Q3.1.2_Diabetes_Linear_ROCAUC.png, Q3.1.2_Diabetes_RBF_ROCAUC.png

***********************************************************************************************************************************************

Q4)

Q4.1, Q4.2 - refer to Question4.py for implementation
	refer to Q4_FinalOutput.txt for detail output; Test Accuracy Mean =  0.9944444444444445
	DS1_Linear.png, DS1_RBF.png, DS2_Linear.png, DS2_RBF.png, DS3_Linear.png, DS3_RBF.png for ROC-AUC curve for each class, across all k-folds.
	
	





























	
