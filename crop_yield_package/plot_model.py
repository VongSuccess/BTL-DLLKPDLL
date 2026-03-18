import matplotlib.pyplot as plt

models = ['Linear Regression', 'Ridge', 'Random Forest', 'XGBoost']
rmse = [7.8, 7.2, 5.9, 5.5]

plt.figure()
plt.bar(models, rmse)

plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Model Comparison (RMSE)')

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig('model_comparison.png')
plt.show()