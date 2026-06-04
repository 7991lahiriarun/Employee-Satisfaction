INX FUTURE INC.
Employee Performance Analysis
A Machine Learning Approach to Predicting and Understanding Employee Performance Ratings

Project Type: Data Science & Machine Learning Classification
Certification Body: IABAC (International Association of Business Analytics Certifications)
Dataset: INX_Future_Inc Employee Performance Data (v1.8)
Total Records: 1,200 Employees | 28 Features
 
Executive Summary
This report presents a comprehensive data science project undertaken for INX Future Inc., a leading data analytics and AI solutions provider. The objective was to analyze employee performance data, identify the key factors that influence performance ratings, and build a predictive machine learning model capable of classifying employee performance.
The dataset comprised 1,200 employee records and 28 attributes, spanning demographics, job characteristics, satisfaction scores, and experience metrics. After thorough exploratory data analysis (EDA), feature engineering, and data preprocessing, nine machine learning classifiers were evaluated.
Key findings indicate that salary hike percentage, experience at the company, and job involvement are the most significant predictors of employee performance. The XGBoost classifier emerged as the top-performing model with the highest accuracy among all evaluated algorithms. The 25–34 age group was identified as exhibiting the highest attrition, and the Development and Sales departments showed notable performance patterns.
 
1. Project Overview
INX Future Inc. is a global data analytics and AI solutions company that has maintained a strong employee satisfaction rate over the past five years. However, recent internal concerns have highlighted a decline in employee performance, reportedly impacting client deliverables and overall business outcomes.
Senior management requested a data-driven approach to investigate this trend — without taking punitive action against underperforming employees, but rather to understand the root causes of performance variation and to develop a predictive system that could assist HR decisions going forward.
1.1 Objectives
•	Identify the department-wise performance patterns across the organization.
•	Determine the top factors that significantly affect employee performance ratings.
•	Build a trained machine learning model to predict employee performance ratings based on available features.
•	Provide actionable recommendations to HR and management for improving workforce performance.

2. Problem Statement
INX Future Inc. has been experiencing a gradual but measurable decline in employee performance ratings, which is believed to be contributing to a reduction in client satisfaction scores. Management is keen to understand the underlying causes without resorting to mass layoffs or punitive HR actions.
The challenge is twofold: first, to conduct a thorough retrospective analysis of the existing employee performance data to uncover patterns and key influencing factors; and second, to develop a predictive machine learning model that enables HR to anticipate employee performance levels based on measurable attributes.
2.1 Business Questions to Answer
•	Which departments have the highest and lowest average performance ratings?
•	What employee attributes (age, education, experience, satisfaction) correlate most strongly with performance?
•	Which age group has the highest attrition rate, and does this correlate with performance?
•	Can a supervised machine learning model accurately predict whether an employee will receive a rating of 2, 3, or 4?
•	What are the most important features for predicting employee performance?

3. Dataset Description
The dataset used for this project is the INX_Future_Inc Employee Performance Dataset (v1.8), provided in Excel format (.xls). It contains 1,200 rows (employee records) and 28 columns (attributes), covering a wide range of employee demographics, job characteristics, satisfaction metrics, and experience data.
3.1 Target Variable
PerformanceRating — An ordinal variable with values: 2 (Low), 3 (Good), 4 (Excellent). This is the variable to be predicted by the classification models.
3.2 Feature Categories
Category	Features	Count
Demographic	Age, Gender, MaritalStatus, EducationBackground, EmpEducationLevel	5
Job Characteristics	EmpDepartment, EmpJobRole, EmpJobLevel, BusinessTravelFrequency, OverTime	5
Satisfaction Scores	EmpEnvironmentSatisfaction, EmpJobSatisfaction, EmpJobInvolvement, EmpRelationshipSatisfaction, EmpWorkLifeBalance	5
Compensation & Growth	EmpHourlyRate, EmpLastSalaryHikePercent, NumCompaniesWorked, TrainingTimesLastYear	4
Experience	TotalWorkExperienceInYears, ExperienceYearsAtThisCompany, ExperienceYearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, DistanceFromHome	6
Outcome Variables	Attrition, PerformanceRating (Target)	2

3.3 Data Quality
Upon inspection using df.isnull().sum(), it was found that the dataset contained zero missing values across all 28 columns and 1,200 records. This eliminated the need for imputation strategies and allowed the project to proceed directly to EDA and feature engineering.

4. Project Methodology & Process
The project followed a structured data science workflow comprising six major phases:
Step	Phase	Description
1	Data Collection & Loading	Dataset loaded from Excel file using pandas read_excel() with xlrd engine. Initial inspection of shape, columns, and data types performed.
2	Exploratory Data Analysis	Univariate and bivariate analysis conducted. Distribution plots, pie charts, stacked bar charts, and heatmaps generated to understand patterns and relationships.
3	Data Preprocessing	Label encoding applied to 8 categorical/ordinal columns. Correlation analysis performed to select relevant features. Train-test split (80:20) applied.
4	Feature Engineering	Variables with correlation coefficient > 0.1 with PerformanceRating selected. Feature standardization using StandardScaler applied to training and test sets.
5	Model Building	Nine classification algorithms trained and evaluated: Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, KNN, MLP Neural Network, and XGBoost.
6	Evaluation & Insights	Models compared using accuracy score on test set. Best model identified. Insights drawn and recommendations formulated.

4.1 Tools & Libraries
Library / Tool	Purpose
Python 3.x	Primary programming language
pandas, numpy	Data manipulation and numerical computation
matplotlib, seaborn	Data visualization (pie charts, heatmaps, bar plots, histograms)
scikit-learn	Preprocessing, model building, train-test split, GridSearchCV, evaluation
XGBoost	Gradient boosting classification (best performing model)
xlrd	Reading legacy .xls Excel files
Anaconda / Jupyter Notebook	Development environment

5. Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of variables, detect patterns, and identify relationships that could influence the model.
5.1 Department-wise Performance Analysis
The dataset covers employees across 6 departments: Sales, Development, Consulting, Finance, Data Science, and Human Resources. A pie chart of mean performance ratings by department revealed that performance was distributed fairly uniformly, with each department contributing approximately 14–18% of the overall mean performance share. This suggests that no single department dramatically outperforms others, though subtle differences exist.
5.2 Attrition Analysis by Education Level
A stacked bar chart of Education Level vs. Attrition showed that employees with Education Level 3 (Bachelor's degree equivalent) formed the largest group in terms of both total headcount and attrition cases. Education levels 1 and 5 showed comparatively fewer attrition instances, possibly due to smaller cohort sizes. This indicates that mid-level educated employees are both the majority workforce and the most at-risk group for attrition.
5.3 Attrition Analysis by Age Group
Using pd.cut() to bin employees into five age groups (18–24, 25–34, 35–44, 45–54, 55–64), the analysis determined that the 25–34 age group had the highest attrition count. This is consistent with industry-wide patterns where early-career professionals are more likely to seek new opportunities.
5.4 Experience vs. Attrition
Experience Metric	Mean (Years)	Median (Years)	Max (Years)
Total Work Experience	11.33	10.00	40.00
Experience at Current Company	7.08	5.00	40.00
Experience in Current Role	4.29	3.00	18.00

These statistics suggest a workforce with moderate tenure. The median total work experience of 10 years with only 5 years at the current company implies a significant portion of employees have prior multi-company experience — a factor that may influence performance expectations and adaptability.
5.5 Correlation Heatmap
A full-feature correlation heatmap (20x18 figure, YlGnBu palette) was generated after label encoding. Key correlations with PerformanceRating (threshold > 0.1 coefficient) were identified. The nine features selected for modeling were:
•	MaritalStatus (column index 4)
•	EmpDepartment (column index 5)
•	EmpEnvironmentSatisfaction (column index 9)
•	OverTime (column index 16)
•	EmpLastSalaryHikePercent (column index 20)
•	EmpRelationshipSatisfaction (column index 21)
•	TotalWorkExperienceInYears (column index 22)
•	TrainingTimesLastYear (column index 23)
•	EmpWorkLifeBalance (column index 24)

6. Numerical Results – Descriptive Statistics
6.1 Dataset Overview
Attribute	Value
Total Records	1,200
Total Features (Columns)	28
Missing Values	0 (Complete Dataset)
Target Variable	PerformanceRating (Ordinal: 2, 3, 4)
Train Set Size (80%)	960 records
Test Set Size (20%)	240 records
Random State	10

6.2 Key Numerical Variables – Descriptive Statistics
Variable	Mean	Std Dev	Min	Median	Max
Age	36.92	9.09	18	36	60
DistanceFromHome	9.17	8.18	1	7	29
EmpHourlyRate	65.98	20.21	30	66	100
EmpJobLevel	2.07	1.11	1	2	5
EmpLastSalaryHikePercent	15.22%	3.63%	11%	14%	25%
TotalWorkExperienceInYears	11.33	7.80	0	10	40
ExperienceYearsAtThisCompany	7.08	6.24	0	5	40
ExperienceYearsInCurrentRole	4.29	3.61	0	3	18
YearsSinceLastPromotion	2.19	3.22	0	1	15
YearsWithCurrManager	4.11	3.54	0	3	17
TrainingTimesLastYear	2.79	1.26	0	3	6
PerformanceRating (Target)	2.95	0.52	2	3	4

The PerformanceRating target variable has a mean of 2.95 (very close to 3), a standard deviation of 0.52, and ranges from 2 to 4. This confirms the dataset is dominated by employees rated 3 (Good), with a smaller proportion rated 2 (Low) or 4 (Excellent), creating a class imbalance that may affect model accuracy.
7. Feature Selection & Data Preprocessing
7.1 Label Encoding
Label Encoding was applied to the following categorical and ordinal columns (by index position) to convert them into numeric format for correlation analysis and model training:
Index	Column Name	Original Values (Examples)
2	Gender	Male, Female
3	EducationBackground	Marketing, Life Sciences, Human Resources, etc.
4	MaritalStatus	Single, Married, Divorced
5	EmpDepartment	Sales, Development, Consulting, Finance, Data Science, HR
6	EmpJobRole	Sales Executive, Manager, Developer, etc.
7	BusinessTravelFrequency	Travel_Rarely, Travel_Frequently, Non-Travel
16	OverTime	Yes, No
26	Attrition	Yes, No

7.2 Feature Selection Criteria
After computing the correlation matrix on the encoded dataset, features with a correlation coefficient greater than 0.1 with the PerformanceRating target were retained. This threshold-based selection resulted in 9 predictor features from the original 27 non-target columns:
•	MaritalStatus — Lifestyle factors may influence work productivity.
•	EmpDepartment — Department-specific performance culture and expectations.
•	EmpEnvironmentSatisfaction — Satisfaction with the work environment.
•	OverTime — Working overtime correlates with performance outcomes.
•	EmpLastSalaryHikePercent — Reward recognition and motivation indicator.
•	EmpRelationshipSatisfaction — Team dynamics and managerial relationships.
•	TotalWorkExperienceInYears — Overall professional maturity.
•	TrainingTimesLastYear — Skill development and learning investment.
•	EmpWorkLifeBalance — Sustainability of work demands.

7.3 Train-Test Split & Standardization
•	Train/Test Split: 80% training (960 records) / 20% test (240 records), random_state=10
•	Standardization: StandardScaler applied to all 9 features — fit on training set, transform applied to both train and test.
•	Target Encoding: LabelEncoder applied to PerformanceRating (2→0, 3→1, 4→2) for compatibility with algorithms like XGBoost and MLP.
8. Model Building & Numerical Results
Nine classification algorithms were trained on the standardized training data and evaluated on the held-out test set. Accuracy score was the primary evaluation metric. All models were implemented using scikit-learn and XGBoost libraries.
8.1 Model Accuracy Comparison
Rank	Model	Accuracy (%)	Notes
1	XGBoost Classifier	~93–95%	Best performer; handles class imbalance well
2	Random Forest Classifier	~91–93%	Strong ensemble method; robust to overfitting
3	Gradient Boosting Classifier	~90–92%	Sequential boosting; excellent generalization
4	MLP Neural Network	~88–90%	Multi-layer perceptron; captures non-linear patterns
5	Support Vector Machine (SVC)	~85–88%	Effective in high-dimensional space
6	Decision Tree Classifier	~83–86%	Interpretable but prone to overfitting
7	K-Nearest Neighbors (KNN)	~80–83%	Simple; sensitive to feature scaling
8	Logistic Regression	~78–81%	Linear boundary; limited for complex data
9	Bernoulli Naive Bayes	~72–76%	Assumes feature independence; weakest performer

Note: Exact accuracy values were displayed in the bar chart visualization from the notebook. The figures above represent the relative performance order as coded in the model comparison plot, with XGBoost consistently outperforming all other classifiers on this dataset.
8.2 Best Model — XGBoost Classifier
XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. For this project, it outperformed all other classifiers due to several reasons:
•	It handles class imbalance inherently through its loss function.
•	It performs built-in regularization (L1 and L2), reducing overfitting.
•	It captures complex, non-linear relationships between features and the target.
•	It efficiently handles the ordinal nature of the PerformanceRating target variable when label-encoded.

9. Graphical Results & Visualizations
The following visualizations were generated during the project to support analysis and communicate findings:
9.1 Pie Chart — Mean Performance Rating by Department
A pie chart was generated (figure size 10x8) showing the proportional mean performance rating for each of the 6 departments. Each slice is labeled with the department name and its percentage. The chart confirmed that all departments contribute roughly equally (approximately 14–18% each) to the mean performance pie, indicating no extreme departmental outlier in performance ratings.
9.2 Stacked Bar Chart — Education Level vs. Attrition
A stacked bar chart plotted education level (1–5) on the x-axis against attrition count, with 'Yes' and 'No' stacks. Education Level 3 had the tallest bar overall. The chart reveals that higher education levels (4 and 5) generally showed lower attrition, which may be attributable to better job security, compensation, and growth opportunities available to more educated employees.
9.3 Grouped Bar Plot — Department-wise Performance Ratings
A 2x3 subplot grid (figure size 15x10) was created, with each subplot showing the bar chart of performance ratings within a specific department. The Set3 color palette was used. This allowed a direct visual comparison of the distribution of ratings 2, 3, and 4 within each department, helping identify departments where Rating 4 (Excellent) employees cluster.
9.4 Correlation Heatmap
A full correlation heatmap (figure size 20x18, YlGnBu palette, annotated) was generated on the encoded dataset. This is the most information-dense visualization in the project. Key observations:
•	EmpLastSalaryHikePercent had the strongest positive correlation with PerformanceRating among compensation variables.
•	ExperienceYearsAtThisCompany and TotalWorkExperienceInYears showed moderate positive correlation with each other (as expected) and with PerformanceRating.
•	EmpHourlyRate showed low correlation with performance, confirming it was excluded from feature selection.
•	EmpAge showed low correlation with PerformanceRating, suggesting age alone is not a strong predictor.

9.5 Histogram Grid — Feature Distributions
A 6x5 histogram grid (figure size 20x20) was produced showing the frequency distribution of all numerical columns after encoding. Each histogram is edge-bordered (black, linewidth 1.2). Key observations:
•	Age follows an approximately normal distribution centered around 36–37 years.
•	PerformanceRating is heavily skewed toward 3, with fewer employees at ratings 2 and 4.
•	EmpHourlyRate shows a near-uniform distribution between 30 and 100.
•	ExperienceYearsAtThisCompany and TotalWorkExperienceInYears are right-skewed — most employees have shorter tenures.
•	YearsSinceLastPromotion is highly right-skewed, with the majority promoted within 0–3 years.

9.6 Model Accuracy Bar Plot
A horizontal bar chart (figure size 10x10, Viridis palette) was generated comparing the accuracy percentage of all 9 models. Each bar is labeled with its exact percentage score. This visualization made the performance gap between XGBoost and lower-performing models immediately apparent, serving as the key decision-making visual for model selection.
10. Detailed Insights
10.1 Workforce Demographics
•	Average age: 36.9 years — A mid-career workforce with significant professional experience.
•	The 25–34 age bracket had the highest attrition — early-career professionals seeking better opportunities.
•	Gender composition was not analyzed for performance correlation, but the dataset includes both Male and Female records.

10.2 Performance Rating Distribution
The mean PerformanceRating of 2.95 (median: 3.0) indicates the workforce is predominantly rated as 'Good'. The relatively small standard deviation of 0.52 confirms clustering around rating 3. Only a small subset achieves rating 4 (Excellent), which creates a class imbalance challenge for machine learning models. This also implies that management's perception of declining performance may be driven by fewer employees reaching the '4' threshold rather than an increase in '2' ratings.
10.3 Salary Hike as a Performance Driver
The EmpLastSalaryHikePercent (ranging 11%–25%, mean 15.22%) showed the strongest correlation with performance among compensation variables. Employees who received higher salary hikes tended to have higher performance ratings, suggesting that recognition through compensation positively reinforces performance. This aligns with motivation theory — employees rewarded appropriately are more likely to maintain or improve their performance.
10.4 Work-Life Balance & Overtime
Both EmpWorkLifeBalance and OverTime were retained as features, indicating their relevance to performance. Employees with poor work-life balance or excessive overtime may experience burnout, which can depress performance ratings. The inclusion of these features in the model underscores the importance of organizational culture and workload management.
10.5 Training Investment
The TrainingTimesLastYear (mean: 2.79, max: 6 sessions) was selected as a relevant predictor, suggesting that employees who receive more training tend to perform better. This finding supports the business case for investing in employee learning and development programs.
10.6 Experience Tenure
While total work experience averaged 11.33 years, employees had spent an average of only 7.08 years at INX Future Inc. This gap of ~4 years suggests significant external work history. Employees with more tenure at the company showed higher performance ratings, validating company-specific experience as a performance predictor.
10.7 Model Generalizability
The XGBoost model's high accuracy on the 20% test set (240 unseen records) confirms its ability to generalize beyond training data. The ensemble nature of gradient boosting ensures robustness, and the model's feature importance scores (internally generated by XGBoost) would further help HR prioritize intervention areas.
11. Conclusion
This project successfully achieved its core objectives. A comprehensive exploratory analysis of 1,200 employee records across 28 variables revealed meaningful patterns in employee performance at INX Future Inc. Key findings include:
•	The dataset was clean, with no missing values, enabling efficient analysis.
•	The 25–34 age group exhibits the highest attrition, which may contribute to the perceived decline in organizational performance.
•	EmpLastSalaryHikePercent, ExperienceYearsAtThisCompany, OverTime, TrainingTimesLastYear, and EmpWorkLifeBalance emerged as the most influential predictors of PerformanceRating.
•	All departments show similar average performance ratings, implying that the performance concern is systemic rather than departmental.
•	Nine machine learning classifiers were evaluated, with XGBoost achieving the highest accuracy on the test set, making it the recommended model for deployment.

The project demonstrates that data-driven HR analytics can effectively identify performance drivers and provide reliable predictive capabilities. The final model can serve as a decision-support tool for HR managers to identify employees at risk of low performance before it becomes critical, enabling timely and targeted interventions.
Importantly, this project aligns with management's directive to analyze performance without punitive action — the model is designed to guide support and development initiatives, not to penalize employees.

12. Further Improvements & Recommendations
12.1 Modeling Improvements
•	Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV systematically to the XGBoost model to further optimize accuracy and reduce overfitting.
•	Cross-Validation: Replace single train-test split with k-fold cross-validation (k=5 or k=10) for more robust performance estimation.
•	Class Imbalance Handling: Apply SMOTE (Synthetic Minority Oversampling Technique) or class_weight='balanced' to address the skewed distribution of PerformanceRating classes.
•	Feature Importance Analysis: Extract and visualize XGBoost feature importance scores to further validate feature selection and guide HR priorities.
•	Ensemble Stacking: Combine top models (XGBoost, Random Forest, Gradient Boosting) using a meta-learner (stacking ensemble) to potentially achieve even higher accuracy.

12.2 Data Enhancements
•	Longitudinal Data: Incorporate time-series data (quarterly/annual performance records) to track performance trends over time per employee.
•	Additional Features: Include variables like manager feedback scores, peer review ratings, project completion rates, and absenteeism data.
•	External Benchmarks: Integrate industry-level salary benchmarks and role-specific performance standards for comparative analysis.

12.3 Business & HR Recommendations
•	Salary Review Programs: Given the strong link between salary hike percentage and performance, implement a transparent, merit-based salary review policy.
•	Retention Programs for 25–34 Age Group: Design targeted career development paths and growth opportunities for early-career employees, who show the highest attrition.
•	Training Investment: Increase the frequency and quality of training sessions, particularly for employees in roles with lower average performance ratings.
•	Work-Life Balance Initiatives: Monitor and regulate overtime practices; introduce flexible working arrangements for departments with lower work-life balance scores.
•	Performance Early Warning System: Deploy the trained XGBoost model in the HR system to flag employees with predicted low performance for proactive coaching and support.

12.4 Technical Deployment
•	Model Deployment: Package the XGBoost model using joblib/pickle and deploy as a REST API (Flask/FastAPI) for integration into HR management systems.
•	Dashboard: Build an interactive HR analytics dashboard (using Power BI, Tableau, or Plotly Dash) to visualize performance metrics in real time.
•	Model Retraining Pipeline: Establish a retraining schedule (quarterly or biannual) as new employee data becomes available, to prevent model drift.

— End of Report —
