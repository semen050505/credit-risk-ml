# Credit Risk Classification

Пет-проект по бинарной классификации заёмщиков на основе датасета German Credit Data. Цель — предсказать, относится ли сумма кредита заёмщика к "высокому" классу (выше медианы) на основе его социально-демографических и финансовых характеристик.

## Что делает ноутбук

1. Загрузка и обзор данных (`pandas`).
2. Заполнение пропусков в признаках `Saving accounts` и `Checking account`.
3. Кодирование категориальных признаков: `OrdinalEncoder` для упорядоченных категорий (`Saving accounts`, `Checking account`), `LabelEncoder` — для остальных (`Sex`, `Housing`, `Purpose`).
4. Визуализация распределений (`seaborn`, `matplotlib`): распределение возраста, суммы кредита, корреляционная матрица.
5. Формирование таргета: `Credit amount > median(Credit amount)`.
6. Стандартизация признаков (`StandardScaler`), разбиение на train/test (80/20).
7. Обучение и сравнение двух моделей:
   - `RandomForestClassifier`
   - `GradientBoostingClassifier` с подбором гиперпараметров через `GridSearchCV` (`n_estimators`, `min_samples_leaf`, `max_depth`)
8. Оценка качества: accuracy, precision, recall, F1, confusion matrix.
9. Сохранение обученной модели (`joblib`).

## Результаты

| Модель | Accuracy |
|---|---|
| Random Forest | 0.750 |
| Gradient Boosting (после GridSearchCV) | 0.765 |

## Стек

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
