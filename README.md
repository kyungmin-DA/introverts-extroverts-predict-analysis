# predict-the-introverts-extroverts-analysis
내향적인 사람과 외향적인 사람 예측 분석 프로젝트(kaggle playground, 2025)

## 1. 프로젝트 개요 (Introduction)

* **분석 배경 및 목적:**
    * 이 프로젝트는 Kaggle의 'Playground Series - S5E7' 컴피티션 참여를 위해 진행되었습니다.
    * 주어진 설문조사 데이터를 바탕으로, 응답자의 성격 유형(내향형/외향형)을 예측하는 머신러닝 모델을 개발하고 성능을 최적화하는 것을 목표로 합니다.
* **분석 기간:** 2025-XX-XX ~ 2025-XX-XX
* **사용 데이터:** [Kaggle Playground Series - S5E7 Dataset](https://www.kaggle.com/competitions/playground-series-s5e7/data)

## 2. 주요 분석 내용 (Key Features)

### 가. 데이터 전처리 (Data Preprocessing)
* 결측치 확인 및 처리 방법을 설명합니다. (예: `SimpleImputer`를 사용한 평균값 대체 등)
* 범주형 변수를 어떻게 수치형으로 변환했는지 설명합니다. (예: 원-핫 인코딩, 라벨 인코딩 등)
* 파생변수를 만들었다면 어떤 변수를 어떻게 만들었는지 설명합니다.

### 나. 탐색적 데이터 분석 (EDA)
* 각 변수(Feature)가 예측 대상(Target)인 성격 유형과 어떤 관계가 있는지 시각화를 통해 보여줍니다.
* **가장 중요하고 흥미로운 발견을 한두 개 정도** 뽑아서 그래프 이미지와 함께 설명합니다.
    * (예시) "분석 결과, '친구들과의 약속' 관련 질문에서 특정 답변을 한 그룹이 외향형일 확률이 월등히 높게 나타났습니다."
    * 
### 다. 모델링 (Modeling)
* 사용한 머신러닝 모델들을 나열합니다. (예: `Logistic Regression`, `RandomForest`, `XGBoost`, `LightGBM` 등)
* 최종적으로 어떤 모델을 선택했는지, 그리고 그 이유는 무엇인지 간략하게 설명합니다. (예: "다양한 모델 중 `XGBoost`가 가장 높은 예측 정확도(Accuracy)를 보여 최종 모델로 선정했습니다.")
* 모델 성능을 어떻게 평가했는지 지표를 함께 보여줍니다. (예: Accuracy, F1-Score, ROC-AUC 등)
    * **Kaggle Public Score / Private Score:** XXX점

## 3. 결론 및 인사이트 (Conclusion & Insights)

* **프로젝트 요약:** 최종 모델의 성능과 분석 결과를 간결하게 요약합니다.
* **인사이트:** 이 프로젝트를 통해 무엇을 배울 수 있었는지, 어떤 점이 아쉬웠는지 등을 자유롭게 작성합니다.
    * (예시) "X 변수가 예측에 가장 중요한 영향을 미치는 것을 확인했으며, Y 피처 엔지니어링을 통해 성능을 0.X% 향상시킬 수 있었습니다. 다만, Z 모델의 하이퍼파라미터 튜닝에 더 많은 시간을 투자했다면 순위를 더 높일 수 있었을 것입니다."

## 4. 사용 기술 (Tech Stack)

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, etc.
* **Tool:** Jupyter Notebook

---
