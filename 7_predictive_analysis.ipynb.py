import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('../data/spacex_launch_data.csv')

# 准备特征和目标
X = df[['launch_site', 'payload_mass', 'launch_year']]
y = df['success']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['payload_mass', 'launch_year']),
        ('cat', OneHotEncoder(), ['launch_site'])
    ]
)

# 模型定义
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

# 超参数网格
param_grids = {
    'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'Random Forest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 5, 10]}
}

# 训练和评估模型
results = {}
best_model = None
best_score = 0

for name, model in models.items():
    # 创建管道
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    # 评估
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 保存结果
    results[name] = {
        'model': grid_search.best_estimator_,
        'accuracy': accuracy,
        'params': grid_search.best_params_
    }

    # 更新最佳模型
    if accuracy > best_score:
        best_score = accuracy
        best_model = name

# 可视化结果
# 1. 模型准确率比较
accuracies = [results[name]['accuracy'] for name in results]
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
plt.savefig('model_accuracy.png')
plt.show()

# 2. 最佳模型混淆矩阵
best_model = results[best_model]['model']
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix for {best_model.named_steps["classifier"].__class__.__name__}')
plt.savefig('confusion_matrix.png')
plt.show()

# 保存最佳模型
import joblib

joblib.dump(best_model, 'best_model.pkl')